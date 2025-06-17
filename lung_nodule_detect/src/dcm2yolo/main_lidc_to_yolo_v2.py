from pathlib import Path
import shutil
import random
import cv2
from collections import Counter
import pydicom

# 导入修订的模块
from project_setup_v2 import setup_project_structure_v2
from xml_parser_v3 import XMLParserV3LIDC
from dicom_processor_v3 import DICOMProcessorV3
from yolo_converter_v2 import YOLOConverterV2


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


def process_lidc_to_yolo_v4_fixed(dicom_root_dir_str, output_root_dir_str,
                                  target_img_size=(512, 512), train_split=0.8,
                                  prefer_ct=True, process_all_modalities=False):
    """
    修正版LIDC到YOLO转换，处理所有序列

    Args:
        prefer_ct: 如果为True，优先处理CT序列
        process_all_modalities: 如果为True，处理所有模态；如果为False，只处理CT
    """
    dicom_root_dir = Path(dicom_root_dir_str)
    dataset_yolo_root = setup_project_structure_v2(output_root_dir_str)

    xml_parser = XMLParserV3LIDC()
    dicom_processor = DICOMProcessorV3()
    yolo_converter = YOLOConverterV2()

    all_yolo_data = []
    modality_stats = Counter()
    strategy_stats = Counter()
    series_processed = Counter()

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

            series_annotations_count = 0

            for xml_file_path in xml_files_to_process:
                print(f"      解析XML: {xml_file_path.name}")

                try:
                    annotations = xml_parser.parse_single_xml(xml_file_path)
                    print(f"        发现 {len(annotations)} 个标注")

                    for ann in annotations:
                        image_sop_uid = ann['image_sop_uid']
                        nodule_id = ann['nodule_id']
                        points = ann['points']

                        # 检查这个SOP UID是否属于当前序列
                        if image_sop_uid in dicom_files_map:
                            target_dcm_path = dicom_files_map[image_sop_uid]

                            print(f"        处理结节: {nodule_id} (SOP: ...{image_sop_uid[-8:]})")

                            # 使用新的处理器
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

                            # 记录统计信息
                            modality_stats[detected_modality] += 1
                            strategy_stats[strategy] += 1
                            series_processed[f"{modality}_{series_id}"] += 1

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
                                'label_content': [yolo_bbox_str],
                                'img_filename_base': img_filename_base,
                                'original_dcm_path': target_dcm_path,
                                'modality': detected_modality,
                                'strategy': strategy,
                                'series_info': f"{study_id_short}.../{series_id}"
                            })

                            series_annotations_count += 1
                            print(f"          ✅ 成功处理: {detected_modality}/{strategy}, bbox: {yolo_bbox_str}")
                        # else:
                        #     print(f"        跳过: SOP UID不属于当前序列")

                except Exception as e:
                    print(f"        ❌ XML解析失败: {e}")

            print(f"   📊 当前序列处理结果: {series_annotations_count} 个标注")

    # 打印最终统计信息
    print(f"\n{'=' * 80}")
    print("🎉 最终处理统计")
    print(f"{'=' * 80}")
    print(f"总处理项目: {len(all_yolo_data)}")

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
            with open(lbl_save_path, 'w') as f:
                for line in item_data['label_content']:
                    f.write(line + "\n")

        print(f"✅ 保存 {len(data_list)} 项到 {split_name} 集合")

    print(f"\n🎉 YOLO数据集生成完成: {dataset_yolo_root}")
    return dataset_yolo_root


if __name__ == "__main__":
    # 配置路径
    LIDC_ROOT = r"D:\python_project\NEU-AI-Integrated-Practical-Training\lung_nodule_detect\Preprocess\LIDC-IDRI-Preprocessing\LIDC-IDRI"
    OUTPUT_ROOT = r"D:\python_project\NEU-AI-Integrated-Practical-Training\lung_nodule_detect\Preprocess\LIDC-IDRI-Preprocessing\LIDC_YOLO_Output_V2_Test"

    if not Path(LIDC_ROOT).exists():
        print(f"❌ 错误: LIDC根目录不存在: {LIDC_ROOT}")
    else:
        print("🎯 选择处理模式：")
        print("1. 只处理CT序列（推荐用于肺结节检测）")
        print("2. 处理所有模态序列")
        print("3. 分析目录结构（不生成数据集）")

        choice = input("请选择 (1/2/3): ").strip()

        if choice == "3":
            # 只分析目录结构
            patient_dirs = sorted([d for d in Path(LIDC_ROOT).iterdir()
                                   if d.is_dir() and d.name.startswith("LIDC-IDRI-")])[:5]  # 分析前5个
            for patient_dir in patient_dirs:
                analyze_patient_structure(patient_dir)
        elif choice == "1":
            print("🔍 只处理CT序列")
            dataset_path = process_lidc_to_yolo_v4_fixed(
                LIDC_ROOT, OUTPUT_ROOT,
                prefer_ct=True, process_all_modalities=False)
        elif choice == "2":
            print("🔍 处理所有模态序列")
            dataset_path = process_lidc_to_yolo_v4_fixed(
                LIDC_ROOT, OUTPUT_ROOT,
                prefer_ct=False, process_all_modalities=True)
        else:
            print("🔍 默认只处理CT序列")
            dataset_path = process_lidc_to_yolo_v4_fixed(
                LIDC_ROOT, OUTPUT_ROOT,
                prefer_ct=True, process_all_modalities=False)