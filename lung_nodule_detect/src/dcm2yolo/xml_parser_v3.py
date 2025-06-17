import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any


class XMLParserV3LIDC:
    """
    处理标准LIDC XML格式的解析器
    支持LidcReadMessage格式的XML文件
    """

    def __init__(self):
        self.namespace = {"ns": "http://www.nih.gov"}  # LIDC XML的命名空间

    def parse_single_xml(self, xml_file_path: Path) -> List[Dict[str, Any]]:
        """
        解析单个LIDC XML文件

        Returns:
            List[Dict]: 每个字典包含:
                - image_sop_uid: 图像的SOP Instance UID
                - nodule_id: 结节ID
                - points: 边界点列表 [(x1, y1), (x2, y2), ...]
                - z_position: Z轴位置（如果有）
                - characteristics: 结节特征（如果需要）
        """
        annotations = []

        try:
            tree = ET.parse(xml_file_path)
            root = tree.getroot()

            print(f"    XML根节点: {root.tag}")

            # 处理命名空间
            if root.tag.startswith('{'):
                # 提取命名空间
                namespace_uri = root.tag.split('}')[0][1:]
                self.namespace = {"ns": namespace_uri}
                root_tag = root.tag.split('}')[1]
            else:
                root_tag = root.tag
                self.namespace = {}

            print(f"    处理 {root_tag} 格式的XML")

            # 根据根节点类型选择解析方法
            if root_tag == 'LidcReadMessage':
                annotations = self._parse_lidc_format(root, xml_file_path)
            elif root_tag == 'IdriReadMessage':
                annotations = self._parse_idri_format(root, xml_file_path)
            else:
                print(f"    ⚠️  未知的XML格式: {root_tag}")
                # 尝试通用解析
                annotations = self._parse_generic_format(root, xml_file_path)

        except ET.ParseError as e:
            print(f"    ❌ XML解析错误: {e}")
        except Exception as e:
            print(f"    ❌ XML处理错误: {e}")
            import traceback
            traceback.print_exc()

        print(f"    解析结果: {len(annotations)} 个标注")
        return annotations

    def _parse_lidc_format(self, root, xml_file_path):
        """解析标准LIDC格式 (LidcReadMessage)"""
        annotations = []

        # 查找所有readingSession
        reading_sessions = self._find_elements(root, 'readingSession')
        print(f"      发现 {len(reading_sessions)} 个阅读会话")

        for session in reading_sessions:
            # 查找unblindedReadNodule
            nodules = self._find_elements(session, 'unblindedReadNodule')
            print(f"        发现 {len(nodules)} 个结节")

            for nodule in nodules:
                # 获取结节ID
                nodule_id_elem = self._find_element(nodule, 'noduleID')
                nodule_id = nodule_id_elem.text if nodule_id_elem is not None else "Unknown Nodule"

                print(f"          处理结节: {nodule_id}")

                # 查找所有ROI
                rois = self._find_elements(nodule, 'roi')
                print(f"            发现 {len(rois)} 个ROI")

                for roi in rois:
                    # 检查inclusion标志
                    inclusion_elem = self._find_element(roi, 'inclusion')
                    if inclusion_elem is not None and inclusion_elem.text.strip().upper() != 'TRUE':
                        print(f"            跳过非包含ROI")
                        continue

                    # 获取图像SOP UID
                    sop_uid_elem = self._find_element(roi, 'imageSOP_UID')
                    if sop_uid_elem is None:
                        print(f"            ⚠️  ROI缺少imageSOP_UID")
                        continue

                    image_sop_uid = sop_uid_elem.text.strip()

                    # 获取Z位置（可选）
                    z_pos_elem = self._find_element(roi, 'imageZposition')
                    z_position = float(z_pos_elem.text.strip()) if z_pos_elem is not None else None

                    # 获取边界点
                    edge_maps = self._find_elements(roi, 'edgeMap')
                    points = []

                    for edge_map in edge_maps:
                        x_elem = self._find_element(edge_map, 'xCoord')
                        y_elem = self._find_element(edge_map, 'yCoord')

                        if x_elem is not None and y_elem is not None:
                            try:
                                x = int(float(x_elem.text.strip()))
                                y = int(float(y_elem.text.strip()))
                                points.append((x, y))
                            except ValueError as e:
                                print(f"            ⚠️  坐标解析错误: {e}")

                    if points:
                        annotation = {
                            'image_sop_uid': image_sop_uid,
                            'nodule_id': nodule_id,
                            'points': points,
                            'z_position': z_position,
                            'source_xml': xml_file_path.name
                        }
                        annotations.append(annotation)
                        print(f"            ✅ 成功解析ROI: {len(points)} 个点, SOP: ...{image_sop_uid[-8:]}")
                    else:
                        print(f"            ⚠️  ROI没有有效的边界点")

        return annotations

    def _parse_idri_format(self, root, xml_file_path):
        """解析IDRI格式 (IdriReadMessage) - 兼容之前的格式"""
        annotations = []

        # 查找CXRreadingSession
        sessions = self._find_elements(root, 'CXRreadingSession')

        for session in sessions:
            reads = self._find_elements(session, 'unblindedRead')

            for read in reads:
                nodule_id_elem = self._find_element(read, 'noduleID')
                nodule_id = nodule_id_elem.text if nodule_id_elem is not None else "Unknown Nodule"

                rois = self._find_elements(read, 'roi')

                for roi in rois:
                    sop_uid_elem = self._find_element(roi, 'imageSOP_UID')
                    if sop_uid_elem is None:
                        continue

                    image_sop_uid = sop_uid_elem.text.strip()

                    edge_maps = self._find_elements(roi, 'edgeMap')
                    points = []

                    for edge_map in edge_maps:
                        x_elem = self._find_element(edge_map, 'xCoord')
                        y_elem = self._find_element(edge_map, 'yCoord')

                        if x_elem is not None and y_elem is not None:
                            try:
                                x = int(float(x_elem.text.strip()))
                                y = int(float(y_elem.text.strip()))
                                points.append((x, y))
                            except ValueError:
                                continue

                    if points:
                        annotations.append({
                            'image_sop_uid': image_sop_uid,
                            'nodule_id': nodule_id,
                            'points': points,
                            'z_position': None,
                            'source_xml': xml_file_path.name
                        })

        return annotations

    def _parse_generic_format(self, root, xml_file_path):
        """通用解析方法，尝试找到常见的结构"""
        annotations = []

        # 尝试找到所有包含imageSOP_UID和edgeMap的元素
        sop_elements = root.findall('.//imageSOP_UID')

        for sop_elem in sop_elements:
            image_sop_uid = sop_elem.text.strip()

            # 找到包含这个SOP UID的父ROI元素
            roi_elem = sop_elem.getparent()

            if roi_elem is not None:
                # 尝试找到结节ID
                nodule_elem = roi_elem.getparent()
                nodule_id = "Generic Nodule"

                if nodule_elem is not None:
                    nodule_id_elem = nodule_elem.find('.//noduleID')
                    if nodule_id_elem is not None:
                        nodule_id = nodule_id_elem.text

                # 获取边界点
                edge_maps = roi_elem.findall('.//edgeMap')
                points = []

                for edge_map in edge_maps:
                    x_elem = edge_map.find('xCoord')
                    y_elem = edge_map.find('yCoord')

                    if x_elem is not None and y_elem is not None:
                        try:
                            x = int(float(x_elem.text.strip()))
                            y = int(float(y_elem.text.strip()))
                            points.append((x, y))
                        except ValueError:
                            continue

                if points:
                    annotations.append({
                        'image_sop_uid': image_sop_uid,
                        'nodule_id': nodule_id,
                        'points': points,
                        'z_position': None,
                        'source_xml': xml_file_path.name
                    })

        return annotations

    def _find_elements(self, parent, tag_name):
        """查找元素，处理命名空间"""
        if self.namespace:
            return parent.findall(f".//ns:{tag_name}", self.namespace)
        else:
            return parent.findall(f".//{tag_name}")

    def _find_element(self, parent, tag_name):
        """查找单个元素，处理命名空间"""
        if self.namespace:
            return parent.find(f".//ns:{tag_name}", self.namespace)
        else:
            return parent.find(f".//{tag_name}")


# 测试代码
if __name__ == "__main__":
    parser = XMLParserV3LIDC()

    # 测试XML文件路径
    test_xml_path = Path(r"你的XML文件路径")

    if test_xml_path.exists():
        print(f"测试解析: {test_xml_path}")
        annotations = parser.parse_single_xml(test_xml_path)

        for i, ann in enumerate(annotations):
            print(f"\n标注 {i + 1}:")
            print(f"  SOP UID: {ann['image_sop_uid']}")
            print(f"  结节ID: {ann['nodule_id']}")
            print(f"  点数量: {len(ann['points'])}")
            if ann['z_position']:
                print(f"  Z位置: {ann['z_position']}")
            print(f"  前5个点: {ann['points'][:5]}")
    else:
        print("请设置正确的XML文件路径进行测试")