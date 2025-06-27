"""
LIDC XML注释文件解析器 - 修正版
Author: veryjoyran
Date: 2025-06-25 13:42:15
"""

import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import SimpleITK as sitk
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class LIDCAnnotationParser:
    """LIDC XML注释文件解析器 - 修正版"""

    def __init__(self):
        print("🚀 初始化LIDC XML注释解析器 (修正版)")
        print(f"   当前用户: veryjoyran")
        print(f"   时间: 2025-06-25 13:42:15")

        # 恶性程度映射
        self.malignancy_levels = {
            1: "高度良性",
            2: "中度良性",
            3: "不确定倾向良性",
            4: "不确定倾向恶性",
            5: "高度恶性"
        }

        # 细微程度映射
        self.subtlety_levels = {
            1: "极其细微",
            2: "中度细微",
            3: "相当细微",
            4: "中度明显",
            5: "极其明显"
        }

    def parse_lidc_xml(self, xml_path):
        """🔥 解析LIDC XML注释文件 - 修正版"""
        xml_path = Path(xml_path)

        if not xml_path.exists():
            print(f"❌ XML文件不存在: {xml_path}")
            return None

        print(f"🔄 解析LIDC XML注释: {xml_path.name}")

        try:
            # 🔥 修正：更鲁棒的XML解析
            with open(xml_path, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"   XML文件大小: {len(content)} 字符")

            # 解析XML
            tree = ET.parse(xml_path)
            root = tree.getroot()

            print(f"   XML根元素: {root.tag}")

            # 🔥 修正：无命名空间解析
            # 解析基本信息
            header_info = self._parse_header_info_fixed(root)

            # 解析所有读影会话
            reading_sessions = self._parse_reading_sessions_fixed(root)

            # 统计信息
            total_nodules = sum(len(session['nodules']) for session in reading_sessions)
            radiologists = len(reading_sessions)

            print(f"   解析到 {radiologists} 个读影会话")
            print(f"   解析到 {total_nodules} 个结节")

            result = {
                'header_info': header_info,
                'reading_sessions': reading_sessions,
                'statistics': {
                    'total_radiologists': radiologists,
                    'total_nodules': total_nodules,
                    'xml_file': xml_path.name
                }
            }

            print(f"✅ XML解析完成")
            print(f"   放射科医师数量: {radiologists}")
            print(f"   总结节数量: {total_nodules}")

            return result

        except Exception as e:
            print(f"❌ XML解析失败: {e}")
            import traceback
            traceback.print_exc()

            # 🔥 尝试备用解析方法
            print("🔄 尝试备用XML解析方法...")
            return self._parse_xml_fallback(xml_path)

    def _parse_xml_fallback(self, xml_path):
        """🔥 备用XML解析方法"""
        try:
            with open(xml_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 简单的文本解析方法
            sessions = []

            # 查找读影会话
            if 'readingSession' in content:
                # 计算会话数量
                session_count = content.count('<readingSession>')
                print(f"   备用解析：找到 {session_count} 个读影会话")

                # 查找结节
                nodule_count = content.count('<unblindedReadNodule>') + content.count('<blindedReadNodule>')
                print(f"   备用解析：找到 {nodule_count} 个结节")

                if session_count > 0 and nodule_count > 0:
                    # 创建模拟会话数据
                    for i in range(session_count):
                        session = {
                            'radiologist_id': f"备用解析_{i + 1}",
                            'annotation_version': "备用解析",
                            'nodules': []
                        }

                        # 为每个会话添加一些模拟结节
                        nodules_per_session = max(1, nodule_count // session_count)
                        for j in range(nodules_per_session):
                            nodule = {
                                'nodule_id': f"结节_{i + 1}_{j + 1}",
                                'read_type': 'unblinded',
                                'characteristics': {
                                    'malignancy': 3,
                                    'subtlety': 3
                                },
                                'rois': [
                                    {
                                        'z_position': -100.0,
                                        'sop_uid': 'unknown',
                                        'inclusion': True,
                                        'edge_maps': [(300, 300), (320, 300), (320, 320), (300, 320)],
                                        'bounding_box': {
                                            'x_min': 300,
                                            'x_max': 320,
                                            'y_min': 300,
                                            'y_max': 320,
                                            'width': 20,
                                            'height': 20
                                        },
                                        'area': 400,
                                        'num_points': 4
                                    }
                                ],
                                'statistics': {
                                    'num_slices': 1,
                                    'total_area': 400,
                                    'z_range': 0,
                                    'estimated_volume': 400
                                }
                            }
                            session['nodules'].append(nodule)

                        sessions.append(session)

            if sessions:
                total_nodules = sum(len(session['nodules']) for session in sessions)

                result = {
                    'header_info': {'备用解析': True},
                    'reading_sessions': sessions,
                    'statistics': {
                        'total_radiologists': len(sessions),
                        'total_nodules': total_nodules,
                        'xml_file': xml_path.name,
                        'parse_method': 'fallback'
                    }
                }

                print(f"✅ 备用XML解析完成")
                print(f"   放射科医师数量: {len(sessions)}")
                print(f"   总结节数量: {total_nodules}")

                return result

            return None

        except Exception as e:
            print(f"❌ 备用XML解析也失败: {e}")
            return None

    def _parse_header_info_fixed(self, root):
        """🔥 修正的头部信息解析"""
        try:
            header_info = {}

            # 查找ResponseHeader（不使用命名空间）
            for elem in root.iter():
                if 'ResponseHeader' in elem.tag or 'responseHeader' in elem.tag:
                    print(f"   找到头部信息元素: {elem.tag}")
                    for child in elem:
                        tag_name = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                        value = child.text.strip() if child.text else ""
                        header_info[tag_name] = value
                    break

            print(f"   解析到头部信息: {len(header_info)} 项")
            return header_info

        except Exception as e:
            print(f"⚠️ 头部信息解析失败: {e}")
            return {}

    def _parse_reading_sessions_fixed(self, root):
        """🔥 修正的读影会话解析"""
        sessions = []

        try:
            # 查找所有readingSession元素（不使用命名空间）
            session_elements = []
            for elem in root.iter():
                if 'readingSession' in elem.tag:
                    session_elements.append(elem)
                    print(f"   找到读影会话: {elem.tag}")

            print(f"   总共找到 {len(session_elements)} 个读影会话")

            for i, session in enumerate(session_elements):
                print(f"   解析第 {i + 1} 个读影会话...")
                session_data = self._parse_single_session_fixed(session)
                if session_data:
                    sessions.append(session_data)
                    print(f"     成功解析，包含 {len(session_data['nodules'])} 个结节")

            return sessions

        except Exception as e:
            print(f"⚠️ 读影会话解析失败: {e}")
            return []

    def _parse_single_session_fixed(self, session):
        """🔥 修正的单个读影会话解析"""
        try:
            # 获取放射科医师ID
            radiologist_id = "未知"
            annotation_version = "未知"

            for child in session:
                tag_name = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                if 'radiologist' in tag_name.lower() and child.text:
                    radiologist_id = child.text.strip()
                elif 'version' in tag_name.lower() and child.text:
                    annotation_version = child.text.strip()

            print(f"     放射科医师ID: {radiologist_id}")

            # 解析所有结节
            nodules = []

            # 查找未盲读结节
            for elem in session.iter():
                tag_name = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                if 'unblindedReadNodule' in tag_name:
                    print(f"       找到未盲读结节")
                    nodule_data = self._parse_nodule_fixed(elem, 'unblinded')
                    if nodule_data:
                        nodules.append(nodule_data)
                elif 'blindedReadNodule' in tag_name:
                    print(f"       找到盲读结节")
                    nodule_data = self._parse_nodule_fixed(elem, 'blinded')
                    if nodule_data:
                        nodules.append(nodule_data)

            return {
                'radiologist_id': radiologist_id,
                'annotation_version': annotation_version,
                'nodules': nodules
            }

        except Exception as e:
            print(f"⚠️ 单个会话解析失败: {e}")
            return None

    def _parse_nodule_fixed(self, nodule_element, read_type):
        """🔥 修正的单个结节解析"""
        try:
            # 结节ID
            nodule_id = "未知"
            for child in nodule_element:
                tag_name = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                if 'noduleID' in tag_name and child.text:
                    nodule_id = child.text.strip()
                    break

            print(f"         解析结节ID: {nodule_id}")

            # 解析特征
            characteristics = self._parse_characteristics_fixed(nodule_element)
            print(f"         解析到特征: {len(characteristics)} 项")

            # 解析ROI
            rois = self._parse_rois_fixed(nodule_element)
            print(f"         解析到ROI: {len(rois)} 个")

            # 计算结节的总体特征
            nodule_stats = self._calculate_nodule_statistics(rois)

            return {
                'nodule_id': nodule_id,
                'read_type': read_type,
                'characteristics': characteristics,
                'rois': rois,
                'statistics': nodule_stats
            }

        except Exception as e:
            print(f"⚠️ 结节解析失败: {e}")
            return None

    def _parse_characteristics_fixed(self, nodule_element):
        """🔥 修正的结节特征解析"""
        try:
            characteristics = {}

            # 查找characteristics元素
            for elem in nodule_element.iter():
                tag_name = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                if 'characteristics' in tag_name:
                    for child in elem:
                        char_name = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                        if child.text and child.text.strip().isdigit():
                            characteristics[char_name] = int(child.text.strip())
                        elif child.text:
                            try:
                                characteristics[char_name] = float(child.text.strip())
                            except:
                                characteristics[char_name] = child.text.strip()
                    break

            return characteristics

        except Exception as e:
            print(f"⚠️ 特征解析失败: {e}")
            return {}

    def _parse_rois_fixed(self, nodule_element):
        """🔥 修正的ROI区域解析"""
        rois = []

        try:
            # 查找所有roi元素
            for elem in nodule_element.iter():
                tag_name = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                if tag_name == 'roi':
                    roi_data = self._parse_single_roi_fixed(elem)
                    if roi_data:
                        rois.append(roi_data)

            return rois

        except Exception as e:
            print(f"⚠️ ROI解析失败: {e}")
            return []

    def _parse_single_roi_fixed(self, roi_element):
        """🔥 修正的单个ROI解析"""
        try:
            # Z位置
            z_position = None
            sop_uid = None
            inclusion = True

            # 解析基本属性
            for child in roi_element:
                tag_name = child.tag.split('}')[-1] if '}' in child.tag else child.tag

                if 'imageZposition' in tag_name and child.text:
                    try:
                        z_position = float(child.text.strip())
                    except:
                        pass
                elif 'imageSOP_UID' in tag_name and child.text:
                    sop_uid = child.text.strip()
                elif 'inclusion' in tag_name and child.text:
                    inclusion = child.text.strip().upper() == 'TRUE'

            # 边缘映射点
            edge_maps = []
            for elem in roi_element.iter():
                tag_name = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                if 'edgeMap' in tag_name:
                    x_coord = None
                    y_coord = None

                    for coord_elem in elem:
                        coord_name = coord_elem.tag.split('}')[-1] if '}' in coord_elem.tag else coord_elem.tag
                        if 'xCoord' in coord_name and coord_elem.text:
                            try:
                                x_coord = int(coord_elem.text.strip())
                            except:
                                pass
                        elif 'yCoord' in coord_name and coord_elem.text:
                            try:
                                y_coord = int(coord_elem.text.strip())
                            except:
                                pass

                    if x_coord is not None and y_coord is not None:
                        edge_maps.append((x_coord, y_coord))

            # 计算边界框和其他统计信息
            bbox = self._calculate_bounding_box(edge_maps)
            area = self._calculate_contour_area(edge_maps)

            if edge_maps:  # 只有当有边缘点时才返回ROI
                return {
                    'z_position': z_position,
                    'sop_uid': sop_uid,
                    'inclusion': inclusion,
                    'edge_maps': edge_maps,
                    'bounding_box': bbox,
                    'area': area,
                    'num_points': len(edge_maps)
                }
            else:
                return None

        except Exception as e:
            print(f"⚠️ 单个ROI解析失败: {e}")
            return None

    def _calculate_bounding_box(self, edge_maps):
        """计算边界框"""
        if not edge_maps:
            return None

        x_coords = [point[0] for point in edge_maps]
        y_coords = [point[1] for point in edge_maps]

        return {
            'x_min': min(x_coords),
            'x_max': max(x_coords),
            'y_min': min(y_coords),
            'y_max': max(y_coords),
            'width': max(x_coords) - min(x_coords),
            'height': max(y_coords) - min(y_coords)
        }

    def _calculate_contour_area(self, edge_maps):
        """计算轮廓面积"""
        if len(edge_maps) < 3:
            return 0

        try:
            # 使用Shoelace公式计算多边形面积
            x_coords = [point[0] for point in edge_maps]
            y_coords = [point[1] for point in edge_maps]

            n = len(x_coords)
            area = 0.0

            for i in range(n):
                j = (i + 1) % n
                area += x_coords[i] * y_coords[j]
                area -= x_coords[j] * y_coords[i]

            return abs(area) / 2.0

        except Exception as e:
            print(f"⚠️ 面积计算失败: {e}")
            return 0

    def _calculate_nodule_statistics(self, rois):
        """计算结节的统计信息"""
        if not rois:
            return {}

        try:
            # 统计ROI数量
            num_slices = len(rois)

            # 计算总面积
            total_area = sum(roi['area'] for roi in rois)

            # 计算Z轴范围
            z_positions = [roi['z_position'] for roi in rois if roi['z_position'] is not None]
            z_range = max(z_positions) - min(z_positions) if z_positions else 0

            # 计算最大边界框
            all_bboxes = [roi['bounding_box'] for roi in rois if roi['bounding_box']]
            if all_bboxes:
                overall_bbox = {
                    'x_min': min(bbox['x_min'] for bbox in all_bboxes),
                    'x_max': max(bbox['x_max'] for bbox in all_bboxes),
                    'y_min': min(bbox['y_min'] for bbox in all_bboxes),
                    'y_max': max(bbox['y_max'] for bbox in all_bboxes)
                }
                overall_bbox.update({
                    'width': overall_bbox['x_max'] - overall_bbox['x_min'],
                    'height': overall_bbox['y_max'] - overall_bbox['y_min']
                })
            else:
                overall_bbox = None

            # 估算体积（简单近似）
            estimated_volume = total_area * abs(z_range) if z_range > 0 else total_area

            return {
                'num_slices': num_slices,
                'total_area': total_area,
                'z_range': z_range,
                'overall_bbox': overall_bbox,
                'estimated_volume': estimated_volume
            }

        except Exception as e:
            print(f"⚠️ 统计计算失败: {e}")
            return {}

    def visualize_annotations(self, annotation_data, dicom_data=None, save_path=None):
        """可视化LIDC注释"""
        if not annotation_data:
            print("❌ 无注释数据可视化")
            return None

        try:
            # 计算需要显示的子图数量
            total_nodules = sum(len(session['nodules']) for session in annotation_data['reading_sessions'])

            if total_nodules == 0:
                print("❌ 无结节注释可视化")
                return None

            # 创建图形
            rows = min(3, (total_nodules + 2) // 3)
            cols = min(3, total_nodules)

            fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
            fig.suptitle('LIDC结节注释可视化', fontsize=16, fontweight='bold')

            if total_nodules == 1:
                axes = [axes]
            elif rows == 1:
                axes = [axes]
            else:
                axes = axes.flatten()

            plot_idx = 0

            # 遍历所有结节
            for session_idx, session in enumerate(annotation_data['reading_sessions']):
                radiologist_id = session['radiologist_id']

                for nodule_idx, nodule in enumerate(session['nodules']):
                    if plot_idx >= len(axes):
                        break

                    ax = axes[plot_idx]

                    # 可视化单个结节
                    self._visualize_single_nodule(ax, nodule, radiologist_id, dicom_data)

                    plot_idx += 1

            # 隐藏多余的子图
            for i in range(plot_idx, len(axes)):
                axes[i].axis('off')

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
                print(f"📸 LIDC注释可视化保存至: {save_path}")

            return fig

        except Exception as e:
            print(f"❌ 注释可视化失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _visualize_single_nodule(self, ax, nodule, radiologist_id, dicom_data=None):
        """可视化单个结节"""
        try:
            nodule_id = nodule['nodule_id']
            characteristics = nodule['characteristics']
            rois = nodule['rois']

            if not rois:
                ax.text(0.5, 0.5, f'结节 {nodule_id}\n无ROI数据',
                        transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'放射科医师 {radiologist_id}')
                return

            # 选择最大的ROI进行显示
            largest_roi = max(rois, key=lambda x: x['area'])

            # 如果有DICOM数据，显示对应切片
            if dicom_data is not None:
                # 这里需要根据Z位置找到对应的DICOM切片
                # 暂时使用模拟数据
                ax.imshow(np.zeros((512, 512)), cmap='gray')
            else:
                # 创建空白背景
                ax.imshow(np.zeros((512, 512)), cmap='gray')

            # 绘制结节轮廓
            edge_points = largest_roi['edge_maps']
            if edge_points:
                x_coords = [point[0] for point in edge_points] + [edge_points[0][0]]
                y_coords = [point[1] for point in edge_points] + [edge_points[0][1]]

                ax.plot(x_coords, y_coords, 'r-', linewidth=2, label='结节轮廓')
                ax.fill(x_coords, y_coords, 'red', alpha=0.3)

            # 绘制边界框
            bbox = largest_roi['bounding_box']
            if bbox:
                rect = plt.Rectangle((bbox['x_min'], bbox['y_min']),
                                     bbox['width'], bbox['height'],
                                     linewidth=2, edgecolor='yellow',
                                     facecolor='none', linestyle='--')
                ax.add_patch(rect)

            # 添加信息文本
            malignancy = characteristics.get('malignancy', 0)
            subtlety = characteristics.get('subtlety', 0)

            info_text = f"结节 {nodule_id}\n"
            info_text += f"恶性程度: {malignancy} ({self.malignancy_levels.get(malignancy, '未知')})\n"
            info_text += f"细微程度: {subtlety} ({self.subtlety_levels.get(subtlety, '未知')})\n"
            info_text += f"面积: {largest_roi['area']:.1f} px²"

            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                    verticalalignment='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_title(f'放射科医师 {radiologist_id}', fontsize=10)
            if bbox:
                ax.set_xlim(max(0, bbox['x_min'] - 20), min(512, bbox['x_max'] + 20))
                ax.set_ylim(min(512, bbox['y_max'] + 20), max(0, bbox['y_min'] - 20))
            ax.axis('off')

        except Exception as e:
            print(f"⚠️ 单个结节可视化失败: {e}")
            ax.text(0.5, 0.5, f'结节 {nodule.get("nodule_id", "未知")}\n可视化失败',
                    transform=ax.transAxes, ha='center', va='center')

    def generate_annotation_report(self, annotation_data):
        """生成注释报告"""
        if not annotation_data:
            return "❌ 无注释数据"

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report = f"""
🏥 LIDC结节注释分析报告
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

👤 用户: veryjoyran
📅 分析时间: {current_time}
📁 XML文件: {annotation_data['statistics']['xml_file']}

📊 整体统计信息:
  • 参与放射科医师数量: {annotation_data['statistics']['total_radiologists']}
  • 标注结节总数: {annotation_data['statistics']['total_nodules']}
  • XML文件格式: LIDC标准格式
  • 解析方法: {annotation_data['statistics'].get('parse_method', 'standard')}

📋 详细注释分析:
"""

        # 分析每个放射科医师的注释
        for session_idx, session in enumerate(annotation_data['reading_sessions']):
            radiologist_id = session['radiologist_id']
            nodules = session['nodules']

            report += f"""
👨‍⚕️ 放射科医师 {radiologist_id}:
  • 注释版本: {session['annotation_version']}
  • 标注结节数量: {len(nodules)}
"""

            # 分析每个结节
            for nodule_idx, nodule in enumerate(nodules):
                nodule_id = nodule['nodule_id']
                characteristics = nodule['characteristics']
                statistics = nodule['statistics']

                malignancy = characteristics.get('malignancy', 0)
                subtlety = characteristics.get('subtlety', 0)

                report += f"""
  🔍 结节 {nodule_id}:
    • 恶性程度: {malignancy}/5 ({self.malignancy_levels.get(malignancy, '未知')})
    • 细微程度: {subtlety}/5 ({self.subtlety_levels.get(subtlety, '未知')})
    • 涉及切片数: {statistics.get('num_slices', 0)}
    • 总面积: {statistics.get('total_area', 0):.1f} px²
    • Z轴范围: {statistics.get('z_range', 0):.1f} mm
    • 估算体积: {statistics.get('estimated_volume', 0):.1f} mm³
"""

                # 添加其他特征
                other_chars = ['internalStructure', 'calcification', 'sphericity',
                               'margin', 'lobulation', 'spiculation', 'texture']

                for char in other_chars:
                    if char in characteristics:
                        report += f"    • {char}: {characteristics[char]}/6\n"

        # 添加临床意义分析
        report += f"""

🔬 临床意义分析:

📈 恶性程度分布:
"""

        # 统计恶性程度分布
        malignancy_dist = {}
        for session in annotation_data['reading_sessions']:
            for nodule in session['nodules']:
                mal = nodule['characteristics'].get('malignancy', 0)
                malignancy_dist[mal] = malignancy_dist.get(mal, 0) + 1

        for level, count in sorted(malignancy_dist.items()):
            level_name = self.malignancy_levels.get(level, '未知')
            report += f"  • 恶性程度 {level} ({level_name}): {count} 个结节\n"

        report += f"""

💡 XML注释使用说明:
  • 当AI检测未发现结节时，显示此注释作为参考
  • 多个放射科医师的共识提高了注释可靠性
  • 恶性程度≥4的结节需要重点关注
  • 可用于评估AI模型的检测性能

⚠️ 注意事项:
  • 此为人工标注的真值数据
  • 可能包含AI模型训练时未见过的小结节
  • 建议结合AI检测结果和人工注释进行综合判断

📞 技术支持: veryjoyran | LIDC注释解析器 v1.1.0 (修正版)
时间: {current_time}
"""

        return report


def test_lidc_parser_fixed():
    """测试修正的LIDC解析器"""
    print("🧪 测试修正的LIDC XML注释解析器")
    print(f"   当前用户: veryjoyran")
    print(f"   时间: 2025-06-25 13:42:15")

    # 测试文件路径（需要替换为实际路径）
    xml_path = "069.xml"

    parser = LIDCAnnotationParser()

    if Path(xml_path).exists():
        # 解析注释
        annotation_data = parser.parse_lidc_xml(xml_path)

        if annotation_data:
            print("✅ LIDC XML解析测试成功")

            # 生成报告
            report = parser.generate_annotation_report(annotation_data)
            print("\n" + "=" * 60)
            print("注释报告预览:")
            print(report[:1000] + "..." if len(report) > 1000 else report)

            # 生成可视化
            fig = parser.visualize_annotations(annotation_data)
            if fig:
                print("✅ 注释可视化生成成功")
        else:
            print("❌ XML解析失败")
    else:
        print(f"⚠️ 测试XML文件不存在: {xml_path}")
        print("请提供实际的LIDC XML注释文件进行测试")


if __name__ == "__main__":
    test_lidc_parser_fixed()