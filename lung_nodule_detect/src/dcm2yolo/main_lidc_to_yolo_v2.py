from pathlib import Path
import shutil
import random
import cv2
from collections import Counter, defaultdict
import pydicom

# 导入修订的模块
from project_setup_v2 import setup_project_structure_v2
from xml_parser_v3 import XMLParserV3LIDC
from dicom_processor_v3 import DICOMProcessorV3
from yolo_converter_v2 import YOLOConverterV2
from lung_segmentation_preprocessor import LungSegmentationPreprocessor


class MultiNoduleProcessor:
    """在原有基础上添加多结节处理功能"""

    def __init__(self):
        self.processed_images = {}  # 缓存已处理的图像

    def collect_annotations_by_image(self, all_annotations):
        """
        按图像SOP UID收集所有标注

        Args:
            all_annotations: 所有解析出的标注列表

        Returns:
            dict: {image_sop_uid: [nodule1, nodule2, ...]}
        """
        image_groups = defaultdict(list)

        for annotation in all_annotations:
            image_sop_uid = annotation['image_sop_uid']
            image_groups[image_sop_uid].append(annotation)

        return dict(image_groups)

    def process_multi_nodule_image(self, image_sop_uid, nodule_list, dicom_files_map,
                                   dicom_processor, yolo_converter, target_img_size,
                                   lung_processor=None, enable_lung_segmentation=False):
        """
        处理包含多个结节的单张图像

        Args:
            lung_processor: 肺分割处理器
            enable_lung_segmentation: 是否启用肺分割
        """
        if image_sop_uid not in dicom_files_map:
            print(f"        ⚠️  图像 SOP UID ...{image_sop_uid[-8:]} 未找到对应DICOM文件")
            return None

        target_dcm_path = dicom_files_map[image_sop_uid]

        # 处理DICOM图像（只处理一次）
        result = dicom_processor.process_dicom_image(target_dcm_path, target_size=target_img_size)

        if len(result) == 4:
            processed_image, original_shape, detected_modality, strategy = result
        else:
            processed_image, original_shape = result[:2]
            detected_modality, strategy = "Unknown", "Legacy"

        if processed_image is None or original_shape is None:
            print(f"        ❌ DICOM处理失败: {target_dcm_path.name}")
            return None

        # 🔥 新增：可选的肺分割后处理
        lung_info = None
        if enable_lung_segmentation and lung_processor and detected_modality == "CT":
            print(f"        🫁 启用肺分割处理...")
            try:
                lung_result = lung_processor.process_8bit_image(processed_image, target_dcm_path.stem)

                if lung_result['success']:
                    processed_image = lung_result['processed_image']  # 使用分割后的图像
                    strategy += "_LungSeg"  # 更新策略标记
                    lung_info = {
                        'lung_mask': lung_result['lung_mask'],
                        'lung_bbox': lung_result['lung_bbox'],
                        'left_lung_mask': lung_result['left_lung_mask'],
                        'right_lung_mask': lung_result['right_lung_mask']
                    }

                    # 计算肺分割统计
                    if lung_result['lung_mask'] is not None:
                        lung_area = lung_result['lung_mask'].sum()
                        total_area = lung_result['lung_mask'].size
                        lung_percentage = (lung_area / total_area) * 100
                        print(f"          肺分割成功: 肺区域占比 {lung_percentage:.1f}%")
                    else:
                        print(f"          肺分割完成但未生成有效掩码")
                else:
                    print(f"          肺分割失败，使用原图: {lung_result.get('processing_notes', [])}")

            except Exception as e:
                print(f"          肺分割处理异常: {e}")

        # 处理该图像中的所有结节
        all_yolo_bboxes = []
        nodule_ids = []

        print(f"        🎯 处理图像 {target_dcm_path.name} 中的 {len(nodule_list)} 个结节")

        for nodule_data in nodule_list:
            nodule_id = nodule_data['nodule_id']
            points = nodule_data['points']

            # 生成YOLO边界框
            yolo_bbox_str = yolo_converter.points_to_yolo_bbox(points, original_shape, target_img_size)

            if yolo_bbox_str:
                all_yolo_bboxes.append(yolo_bbox_str)
                nodule_ids.append(nodule_id)
                print(f"          ✅ {nodule_id}: {yolo_bbox_str}")
            else:
                print(f"          ❌ {nodule_id}: 边界框生成失败")

        if not all_yolo_bboxes:
            print(f"        ⚠️  图像中没有有效的结节标注")
            return None

        # 创建聚合的文件名
        nodule_summary = "_".join([nid.replace(" ", "").replace("Nodule", "N") for nid in nodule_ids])
        if len(nodule_summary) > 30:  # 避免文件名过长
            nodule_summary = f"{len(nodule_ids)}nodules"

        sop_uid_short = image_sop_uid.split('.')[-1][:8]

        return {
            'image_array': processed_image,
            'label_content': all_yolo_bboxes,  # 多个YOLO标注
            'base_filename': f"Multi_{sop_uid_short}_{nodule_summary}_{detected_modality}",
            'original_dcm_path': target_dcm_path,
            'modality': detected_modality,
            'strategy': strategy,
            'nodule_count': len(nodule_ids),
            'nodule_ids': nodule_ids,
            'lung_info': lung_info  # 新增肺分割信息
        }


def analyze_patient_structure(patient_dir):
    """分析病人目录结构，识别不同的Study和Series"""
    patient_path = Path(patient_dir)
    structure_info = []

    print(f"\n🔍 分析病人目录结构: {patient_path.name}")
    print("=" * 60)

    study_dirs = sorted([d for d in patient_path.iterdir()
                         if d.is_dir() and not d.name.lower() == "accuimage.dir"])

    for study_idx, study_dir in enumerate(study_dirs):
        study_id = study_dir.name
        study_id_short = study_id.split('.')[-1][:12]
        print(f"\nStudy {study_idx + 1}: ...{study_id_short}")

        series_dirs = sorted([d for d in study_dir.iterdir()
                              if d.is_dir() and not d.name.lower() == "accuimage.dir"])

        for series_idx, series_dir in enumerate(series_dirs):
            series_id = series_dir.name
            dcm_files = list(series_dir.glob("*.dcm"))
            xml_files = list(series_dir.glob("*.xml"))

            series_info = {
                'study_dir': study_dir,
                'series_dir': series_dir,
                'study_id': study_id,
                'series_id': series_id,
                'dcm_count': len(dcm_files),
                'xml_count': len(xml_files),
                'modality': 'UNKNOWN',
                'series_description': 'Unknown',
                'is_ct': False,
                'hu_range': None
            }

            if dcm_files:
                try:
                    # 读取第一个DICOM文件获取基本信息
                    sample_dcm = pydicom.dcmread(dcm_files[0], stop_before_pixels=False)
                    series_info['modality'] = getattr(sample_dcm, 'Modality', 'UNKNOWN')
                    series_info['series_description'] = getattr(sample_dcm, 'SeriesDescription', 'No Description')

                    # 检查是否为CT
                    if series_info['modality'] == 'CT':
                        series_info['is_ct'] = True
                        # 计算HU范围
                        pixel_array = sample_dcm.pixel_array
                        intercept = float(getattr(sample_dcm, 'RescaleIntercept', 0))
                        slope = float(getattr(sample_dcm, 'RescaleSlope', 1))
                        hu_array = pixel_array * slope + intercept
                        series_info['hu_range'] = (hu_array.min(), hu_array.max())

                    print(f"  Series {series_idx + 1} ({series_id}): {series_info['modality']} | "
                          f"{len(dcm_files)} DCM | {len(xml_files)} XML")
                    print(f"    描述: {series_info['series_description']}")

                    if series_info['is_ct']:
                        print(
                            f"    ✅ CT序列 - HU范围: [{series_info['hu_range'][0]:.0f}, {series_info['hu_range'][1]:.0f}]")
                    else:
                        print(f"    ❌ 非CT序列 ({series_info['modality']})")

                except Exception as e:
                    print(f"  Series {series_idx + 1} ({series_id}): 读取失败 - {e}")

            structure_info.append(series_info)

    return structure_info


def process_lidc_to_yolo_v5_with_lung_seg(dicom_root_dir_str, output_root_dir_str,
                                          target_img_size=(512, 512), train_split=0.8,
                                          prefer_ct=True, process_all_modalities=False,
                                          enable_multi_nodule=True, enable_lung_segmentation=False):
    """
    V5版本：支持多结节聚合处理 + 可选肺分割

    Args:
        enable_multi_nodule: 是否启用多结节聚合处理
        enable_lung_segmentation: 是否启用肺分割预处理
    """
    dicom_root_dir = Path(dicom_root_dir_str)
    dataset_yolo_root = setup_project_structure_v2(output_root_dir_str)

    xml_parser = XMLParserV3LIDC()
    dicom_processor = DICOMProcessorV3()
    yolo_converter = YOLOConverterV2()
    multi_processor = MultiNoduleProcessor()

    # 🔥 新增：可选的肺分割处理器
    lung_processor = None
    if enable_lung_segmentation:
        lung_processor = LungSegmentationPreprocessor()
        # 根据需要可以启用调试模式
        # lung_processor.enable_debug(save_images=True, output_dir="lung_debug", show_comparison=False)
        print("🫁 肺分割模块已启用")

    all_yolo_data = []
    modality_stats = Counter()
    strategy_stats = Counter()
    series_processed = Counter()

    # 增强统计信息
    processing_stats = {
        'total_images': 0,
        'multi_nodule_images': 0,
        'single_nodule_images': 0,
        'total_nodules': 0,
        'lung_segmented_images': 0,  # 新增
        'lung_segmentation_success_rate': 0,  # 新增
        'skipped_duplicate_images': 0
    }

    patient_dirs = sorted([d for d in dicom_root_dir.iterdir()
                           if d.is_dir() and d.name.startswith("LIDC-IDRI-")])
    print(f"Found {len(patient_dirs)} patient directories.")

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        print(f"\n{'=' * 60}")
        print(f"Processing Patient: {patient_id}")
        print(f"{'=' * 60}")

        # 分析病人目录结构
        structure_info = analyze_patient_structure(patient_dir)

        # 根据设置决定处理哪些序列
        series_to_process = []

        if prefer_ct and not process_all_modalities:
            # 只处理CT序列
            ct_series = [info for info in structure_info if info['is_ct']]
            if ct_series:
                series_to_process = ct_series
                print(f"\n🎯 发现 {len(ct_series)} 个CT序列，将只处理CT序列")
                if enable_lung_segmentation:
                    print(f"   🫁 CT序列将应用肺分割预处理")
            else:
                print(f"\n⚠️  未发现CT序列，将处理所有序列")
                series_to_process = structure_info
        else:
            # 处理所有序列
            series_to_process = structure_info
            print(f"\n🎯 将处理所有 {len(series_to_process)} 个序列")

        # 处理选定的序列
        for series_info in series_to_process:
            study_dir = series_info['study_dir']
            series_dir = series_info['series_dir']
            study_id = series_info['study_id']
            series_id = series_info['series_id']
            modality = series_info['modality']

            print(f"\n📁 处理序列: {study_id.split('.')[-1][:12]}.../{series_id} ({modality})")

            # 显示处理配置
            config_info = []
            if enable_multi_nodule:
                config_info.append("多结节聚合")
            if enable_lung_segmentation and modality == "CT":
                config_info.append("肺分割")

            if config_info:
                print(f"   🔧 处理配置: {' + '.join(config_info)}")

            # 构建DICOM文件映射（SOP UID -> 文件路径）
            dicom_files_map = {}
            dcm_files = list(series_dir.glob("*.dcm"))

            print(f"   扫描 {len(dcm_files)} 个DICOM文件...")
            for dcm_file_path in dcm_files:
                sop_uid = dicom_processor.get_sop_uid(dcm_file_path)
                if sop_uid:
                    dicom_files_map[sop_uid] = dcm_file_path

            print(f"   成功映射 {len(dicom_files_map)} 个DICOM文件")

            if not dicom_files_map:
                print(f"   ⚠️  跳过：没有有效的DICOM文件")
                continue

            # 处理XML文件（当前目录和父目录）
            xml_files_to_process = []

            # 1. 当前series目录中的XML
            current_xml_files = list(series_dir.glob("*.xml"))
            xml_files_to_process.extend(current_xml_files)

            # 2. 父study目录中的XML（如果存在）
            parent_xml_files = list(study_dir.glob("*.xml"))
            xml_files_to_process.extend(parent_xml_files)

            # 3. 祖父patient目录中的XML（如果存在）
            grandparent_xml_files = list(patient_dir.glob("*.xml"))
            xml_files_to_process.extend(grandparent_xml_files)

            # 去重
            xml_files_to_process = list(set(xml_files_to_process))

            print(f"   找到 {len(xml_files_to_process)} 个XML文件")

            # 收集所有XML标注
            all_series_annotations = []

            for xml_file_path in xml_files_to_process:
                print(f"      解析XML: {xml_file_path.name}")

                try:
                    annotations = xml_parser.parse_single_xml(xml_file_path)

                    # 只保留属于当前序列的标注
                    series_annotations = [ann for ann in annotations
                                          if ann['image_sop_uid'] in dicom_files_map]
                    all_series_annotations.extend(series_annotations)

                    print(f"        发现 {len(annotations)} 个标注，其中 {len(series_annotations)} 个属于当前序列")

                except Exception as e:
                    print(f"        ❌ XML解析失败: {e}")

            if not all_series_annotations:
                print(f"      ⚠️  没有找到该序列的有效标注")
                continue

            # 根据是否启用多结节处理选择不同的处理方式
            if enable_multi_nodule:
                # 多结节聚合处理
                image_groups = multi_processor.collect_annotations_by_image(all_series_annotations)

                print(
                    f"      📊 多结节处理模式: {len(all_series_annotations)} 个结节分布在 {len(image_groups)} 张图像中")

                # 统计多结节图像
                multi_nodule_count = sum(1 for nodules in image_groups.values() if len(nodules) > 1)
                single_nodule_count = len(image_groups) - multi_nodule_count

                print(f"        - 单结节图像: {single_nodule_count}")
                print(f"        - 多结节图像: {multi_nodule_count}")

                lung_segmentation_attempts = 0
                lung_segmentation_successes = 0

                # 处理每个图像组
                for image_sop_uid, nodule_list in image_groups.items():
                    result = multi_processor.process_multi_nodule_image(
                        image_sop_uid, nodule_list, dicom_files_map,
                        dicom_processor, yolo_converter, target_img_size,
                        lung_processor, enable_lung_segmentation
                    )

                    if result:
                        # 统计肺分割情况
                        if enable_lung_segmentation and result['modality'] == "CT":
                            lung_segmentation_attempts += 1
                            if result.get('lung_info') and result['lung_info'].get('lung_mask') is not None:
                                lung_segmentation_successes += 1
                                processing_stats['lung_segmented_images'] += 1

                        # 转换为统一的数据格式
                        unified_data = {
                            'image_array': result['image_array'],
                            'label_content': result['label_content'],
                            'img_filename_base': result['base_filename'],
                            'original_dcm_path': result['original_dcm_path'],
                            'modality': result['modality'],
                            'strategy': result['strategy'],
                            'series_info': f"{study_id.split('.')[-1][:8]}.../{series_id}",
                            'nodule_count': result['nodule_count'],
                            'lung_info': result.get('lung_info')  # 新增肺分割信息
                        }

                        all_yolo_data.append(unified_data)

                        # 更新统计
                        processing_stats['total_images'] += 1
                        processing_stats['total_nodules'] += result['nodule_count']
                        modality_stats[result['modality']] += result['nodule_count']
                        strategy_stats[result['strategy']] += 1
                        series_processed[f"{modality}_{series_id}"] += result['nodule_count']

                        if result['nodule_count'] > 1:
                            processing_stats['multi_nodule_images'] += 1
                        else:
                            processing_stats['single_nodule_images'] += 1

                        print(f"          ✅ 聚合处理成功: {result['nodule_count']} 个结节")

                # 计算当前序列的肺分割成功率
                if lung_segmentation_attempts > 0:
                    current_success_rate = (lung_segmentation_successes / lung_segmentation_attempts) * 100
                    print(
                        f"      🫁 当前序列肺分割成功率: {current_success_rate:.1f}% ({lung_segmentation_successes}/{lung_segmentation_attempts})")

            else:
                # 原有的单结节处理方式（保持兼容）
                print(f"      📊 单结节处理模式: 处理 {len(all_series_annotations)} 个标注")

                series_annotations_count = 0
                lung_segmentation_attempts = 0
                lung_segmentation_successes = 0

                for ann in all_series_annotations:
                    image_sop_uid = ann['image_sop_uid']
                    nodule_id = ann['nodule_id']
                    points = ann['points']

                    target_dcm_path = dicom_files_map[image_sop_uid]

                    print(f"        处理结节: {nodule_id} (SOP: ...{image_sop_uid[-8:]})")

                    # 使用处理器
                    result = dicom_processor.process_dicom_image(
                        target_dcm_path, target_size=target_img_size)

                    if len(result) == 4:  # 新版本返回4个值
                        processed_image, original_shape, detected_modality, strategy = result
                    else:  # 兼容旧版本
                        processed_image, original_shape = result[:2]
                        detected_modality, strategy = modality, "Legacy"

                    if processed_image is None or original_shape is None:
                        print(f"          ❌ DICOM处理失败")
                        continue

                    # 🔥 新增：可选的肺分割后处理
                    lung_info = None
                    if enable_lung_segmentation and lung_processor and detected_modality == "CT":
                        print(f"          🫁 启用肺分割处理...")
                        lung_segmentation_attempts += 1
                        try:
                            lung_result = lung_processor.process_8bit_image(processed_image, target_dcm_path.stem)

                            if lung_result['success']:
                                processed_image = lung_result['processed_image']
                                strategy += "_LungSeg"
                                lung_info = {
                                    'lung_mask': lung_result['lung_mask'],
                                    'lung_bbox': lung_result['lung_bbox'],
                                    'left_lung_mask': lung_result['left_lung_mask'],
                                    'right_lung_mask': lung_result['right_lung_mask']
                                }
                                lung_segmentation_successes += 1
                                processing_stats['lung_segmented_images'] += 1
                                print(f"            肺分割成功")
                            else:
                                print(f"            肺分割失败，使用原图")

                        except Exception as e:
                            print(f"            肺分割处理异常: {e}")

                    # 记录统计信息
                    modality_stats[detected_modality] += 1
                    strategy_stats[strategy] += 1
                    series_processed[f"{modality}_{series_id}"] += 1
                    processing_stats['total_nodules'] += 1
                    processing_stats['single_nodule_images'] += 1

                    # 生成YOLO边界框
                    yolo_bbox_str = yolo_converter.points_to_yolo_bbox(
                        points, original_shape, target_img_size)

                    if not yolo_bbox_str:
                        print(f"          ❌ 无法生成边界框")
                        continue

                    # 创建唯一文件名（包含序列信息）
                    sop_uid_short = image_sop_uid.split('.')[-1][:8]
                    study_id_short = study_id.split('.')[-1][:8]
                    img_filename_base = (f"{patient_id}_{study_id_short}_{series_id}_"
                                         f"{sop_uid_short}_{nodule_id.replace(' ', '_')}_"
                                         f"{detected_modality}")

                    all_yolo_data.append({
                        'image_array': processed_image,
                        'label_content': [yolo_bbox_str],  # 单个YOLO标注，但仍然用列表格式
                        'img_filename_base': img_filename_base,
                        'original_dcm_path': target_dcm_path,
                        'modality': detected_modality,
                        'strategy': strategy,
                        'series_info': f"{study_id_short}.../{series_id}",
                        'nodule_count': 1,
                        'lung_info': lung_info  # 新增肺分割信息
                    })

                    series_annotations_count += 1
                    print(f"          ✅ 成功处理: {detected_modality}/{strategy}, bbox: {yolo_bbox_str}")

                processing_stats['total_images'] += series_annotations_count

                # 计算当前序列的肺分割成功率
                if lung_segmentation_attempts > 0:
                    current_success_rate = (lung_segmentation_successes / lung_segmentation_attempts) * 100
                    print(
                        f"      🫁 当前序列肺分割成功率: {current_success_rate:.1f}% ({lung_segmentation_successes}/{lung_segmentation_attempts})")

                print(f"   📊 当前序列处理结果: {series_annotations_count} 个标注")

    # 计算总体肺分割成功率
    if processing_stats['lung_segmented_images'] > 0:
        processing_stats['lung_segmentation_success_rate'] = (
                processing_stats['lung_segmented_images'] / processing_stats['total_images'] * 100
        )

    # 打印最终统计信息
    print(f"\n{'=' * 80}")
    print("🎉 LIDC到YOLO转换完成统计")
    if enable_multi_nodule:
        print("   📊 多结节聚合处理模式")
    if enable_lung_segmentation:
        print("   🫁 肺分割预处理模式")
    print(f"{'=' * 80}")
    print(f"总处理项目: {len(all_yolo_data)}")
    print(f"处理图像总数: {processing_stats['total_images']}")
    print(f"结节总数: {processing_stats['total_nodules']}")

    if enable_multi_nodule:
        print(f"单结节图像: {processing_stats['single_nodule_images']}")
        print(f"多结节图像: {processing_stats['multi_nodule_images']}")
        avg_nodules = processing_stats['total_nodules'] / max(1, processing_stats['total_images'])
        print(f"平均每图像结节数: {avg_nodules:.2f}")

    if enable_lung_segmentation:
        print(f"肺分割处理图像: {processing_stats['lung_segmented_images']}")
        if processing_stats['total_images'] > 0:
            lung_seg_rate = (processing_stats['lung_segmented_images'] / processing_stats['total_images']) * 100
            print(f"肺分割成功率: {lung_seg_rate:.1f}%")

    print(f"\n📊 模态分布:")
    for modality, count in modality_stats.most_common():
        print(f"  {modality}: {count}")

    print(f"\n🔧 处理策略分布:")
    for strategy, count in strategy_stats.most_common():
        print(f"  {strategy}: {count}")

    print(f"\n📁 序列处理分布:")
    for series_info, count in series_processed.most_common():
        print(f"  {series_info}: {count}")

    if not all_yolo_data:
        print("\n❌ 没有数据可保存。")
        return None

    # 数据分割和保存
    random.shuffle(all_yolo_data)
    split_idx = int(len(all_yolo_data) * train_split)
    train_data = all_yolo_data[:split_idx]
    val_data = all_yolo_data[split_idx:]

    print(f"\n💾 数据集分割: Train={len(train_data)}, Val={len(val_data)}")

    for split_name, data_list in [("train", train_data), ("val", val_data)]:
        img_dir = dataset_yolo_root / "images" / split_name
        lbl_dir = dataset_yolo_root / "labels" / split_name

        for item_idx, item_data in enumerate(data_list):
            img_filename = f"{item_data['img_filename_base']}_{item_idx}.png"
            lbl_filename = f"{item_data['img_filename_base']}_{item_idx}.txt"

            img_save_path = img_dir / img_filename
            lbl_save_path = lbl_dir / lbl_filename

            cv2.imwrite(str(img_save_path), item_data['image_array'])

            # 支持多个YOLO标注的保存
            with open(lbl_save_path, 'w') as f:
                for yolo_bbox in item_data['label_content']:
                    f.write(yolo_bbox + "\n")

        print(f"✅ 保存 {len(data_list)} 项到 {split_name} 集合")

    print(f"\n🎉 YOLO数据集生成完成: {dataset_yolo_root}")
    return dataset_yolo_root


if __name__ == "__main__":
    # 配置路径
    LIDC_ROOT = r"D:\python_project\NEU-AI-Integrated-Practical-Training\lung_nodule_detect\Preprocess\LIDC-IDRI-Preprocessing\LIDC-IDRI"
    OUTPUT_ROOT = r"D:\python_project\NEU-AI-Integrated-Practical-Training\lung_nodule_detect\Preprocess\LIDC-IDRI-Preprocessing\LIDC_YOLO_Output_V5_Test_NS"

    if not Path(LIDC_ROOT).exists():
        print(f"❌ 错误: LIDC根目录不存在: {LIDC_ROOT}")
    else:
        print("🎯 选择处理模式：")
        print("1. 只处理CT序列（推荐用于肺结节检测）")
        print("2. 处理所有模态序列")
        print("3. 分析目录结构（不生成数据集）")
        print("4. 只处理CT序列 + 多结节聚合")
        print("5. 只处理CT序列 + 肺分割预处理")
        print("6. 只处理CT序列 + 多结节聚合 + 肺分割预处理（完整功能）")

        choice = input("请选择 (1/2/3/4/5/6): ").strip()

        if choice == "3":
            # 只分析目录结构
            patient_dirs = sorted([d for d in Path(LIDC_ROOT).iterdir()
                                   if d.is_dir() and d.name.startswith("LIDC-IDRI-")])[:5]  # 分析前5个
            for patient_dir in patient_dirs:
                analyze_patient_structure(patient_dir)

        elif choice == "1":
            print("🔍 只处理CT序列")
            dataset_path = process_lidc_to_yolo_v5_with_lung_seg(
                LIDC_ROOT, OUTPUT_ROOT,
                prefer_ct=True, process_all_modalities=False,
                enable_multi_nodule=False, enable_lung_segmentation=False)

        elif choice == "2":
            print("🔍 处理所有模态序列")
            dataset_path = process_lidc_to_yolo_v5_with_lung_seg(
                LIDC_ROOT, OUTPUT_ROOT,
                prefer_ct=False, process_all_modalities=True,
                enable_multi_nodule=False, enable_lung_segmentation=False)

        elif choice == "4":
            print("🔍 只处理CT序列 + 多结节聚合")
            dataset_path = process_lidc_to_yolo_v5_with_lung_seg(
                LIDC_ROOT, OUTPUT_ROOT,
                prefer_ct=True, process_all_modalities=False,
                enable_multi_nodule=True, enable_lung_segmentation=False)

        elif choice == "5":
            print("🔍 只处理CT序列 + 肺分割预处理")
            dataset_path = process_lidc_to_yolo_v5_with_lung_seg(
                LIDC_ROOT, OUTPUT_ROOT,
                prefer_ct=True, process_all_modalities=False,
                enable_multi_nodule=False, enable_lung_segmentation=True)

        elif choice == "6":
            print("🔍 只处理CT序列 + 多结节聚合 + 肺分割预处理（完整功能）")
            dataset_path = process_lidc_to_yolo_v5_with_lung_seg(
                LIDC_ROOT, OUTPUT_ROOT,
                prefer_ct=True, process_all_modalities=False,
                enable_multi_nodule=True, enable_lung_segmentation=True)

        else:
            print("🔍 默认只处理CT序列")
            dataset_path = process_lidc_to_yolo_v5_with_lung_seg(
                LIDC_ROOT, OUTPUT_ROOT,
                prefer_ct=True, process_all_modalities=False,
                enable_multi_nodule=False, enable_lung_segmentation=False)