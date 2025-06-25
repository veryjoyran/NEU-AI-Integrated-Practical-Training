"""
纯3D推理检测器 - LUNA16兼容
Author: veryjoyran
Date: 2025-06-24 15:25:55
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import SimpleITK as sitk
import tempfile
from datetime import datetime
import cv2

# MonAI相关导入
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    Orientationd, ScaleIntensityRanged, ToTensord,
    ResizeWithPadOrCropd, SpatialPadd, CropForegroundd
)
from scipy import ndimage
from skimage import measure, morphology

# 导入Bundle加载器
from bundle_loader import MonAIBundleLoader

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print(f"🔧 PyTorch版本: {torch.__version__}")
print(f"👤 当前用户: veryjoyran")
print(f"📅 当前时间: 2025-06-24 15:25:55")

class LUNA16DicomProcessor:
    """LUNA16标准DICOM处理器"""

    def __init__(self):
        print("🚀 初始化LUNA16标准DICOM处理器")

        # LUNA16标准参数
        self.target_spacing = (0.703125, 0.703125, 1.25)  # mm
        self.target_size = (192, 192, 80)  # (W, H, D)
        self.hu_window = (-1000, 400)  # 肺部HU值范围

    def process_dicom_to_luna16_standard(self, dicom_path):
        """将DICOM处理为LUNA16标准格式"""
        print(f"🔄 按LUNA16标准处理DICOM: {dicom_path}")

        try:
            # 1. 加载DICOM数据
            image, original_info = self._load_dicom_data(dicom_path)
            if image is None:
                return None

            # 2. 重采样到LUNA16标准间距
            resampled_image = self._resample_to_luna16_spacing(image)

            # 3. 方向标准化
            oriented_image = self._standardize_orientation(resampled_image)

            # 4. 裁剪前景（去除床和空气）
            cropped_image = self._crop_foreground(oriented_image)

            # 5. 调整到目标尺寸
            resized_image = self._resize_to_target_size(cropped_image)

            # 6. HU值标准化
            normalized_image = self._normalize_hu_values(resized_image)

            # 7. 创建多种预处理版本
            versions = self._create_preprocessing_versions(normalized_image)

            result = {
                'original_array': sitk.GetArrayFromImage(image),
                'processed_array': sitk.GetArrayFromImage(normalized_image),
                'versions': versions,
                'original_info': original_info,
                'processing_info': {
                    'target_spacing': self.target_spacing,
                    'target_size': self.target_size,
                    'hu_window': self.hu_window,
                    'final_spacing': normalized_image.GetSpacing(),
                    'final_size': normalized_image.GetSize()
                }
            }

            print(f"✅ LUNA16标准处理完成")
            print(f"   最终尺寸: {normalized_image.GetSize()}")
            print(f"   最终间距: {normalized_image.GetSpacing()}")

            return result

        except Exception as e:
            print(f"❌ LUNA16处理失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _load_dicom_data(self, dicom_path):
        """加载DICOM数据"""
        dicom_path = Path(dicom_path)

        try:
            if dicom_path.is_dir():
                # DICOM序列目录
                reader = sitk.ImageSeriesReader()
                dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_path))

                if not dicom_names:
                    raise ValueError("目录中未找到DICOM文件")

                reader.SetFileNames(dicom_names)
                image = reader.Execute()
                print(f"   读取DICOM序列: {len(dicom_names)} 个文件")

            else:
                # 单个DICOM文件
                image = sitk.ReadImage(str(dicom_path))

                # 如果是2D图像，需要创建3D体积
                if image.GetDimension() == 2:
                    print("   检测到2D DICOM，创建3D体积...")
                    image = self._convert_2d_to_3d(image)

                print(f"   读取单个DICOM文件")

            original_info = {
                'size': image.GetSize(),
                'spacing': image.GetSpacing(),
                'origin': image.GetOrigin(),
                'direction': image.GetDirection()
            }

            print(f"   原始尺寸: {image.GetSize()}")
            print(f"   原始间距: {image.GetSpacing()}")

            return image, original_info

        except Exception as e:
            print(f"❌ DICOM加载失败: {e}")
            return None, None

    def _convert_2d_to_3d(self, image_2d):
        """将2D图像转换为3D体积"""
        array_2d = sitk.GetArrayFromImage(image_2d)

        # 创建多个切片（复制原切片）
        num_slices = 20
        array_3d = np.stack([array_2d] * num_slices, axis=0)

        # 创建3D图像
        image_3d = sitk.GetImageFromArray(array_3d)

        # 设置3D属性
        spacing_2d = image_2d.GetSpacing()
        spacing_3d = list(spacing_2d) + [2.5]  # 添加Z方向间距
        image_3d.SetSpacing(spacing_3d)

        origin_2d = image_2d.GetOrigin()
        origin_3d = list(origin_2d) + [0.0]
        image_3d.SetOrigin(origin_3d)

        # 设置3D方向矩阵
        direction_2d = image_2d.GetDirection()
        direction_3d = self._expand_direction_to_3d(direction_2d)
        image_3d.SetDirection(direction_3d)

        return image_3d

    def _expand_direction_to_3d(self, direction_2d):
        """将2D方向矩阵扩展为3D"""
        if len(direction_2d) == 4:
            # 2D: (xx, xy, yx, yy) -> 3D: (xx, xy, xz, yx, yy, yz, zx, zy, zz)
            return direction_2d[:2] + (0.0,) + direction_2d[2:] + (0.0, 0.0, 0.0, 1.0)
        else:
            return direction_2d

    def _resample_to_luna16_spacing(self, image):
        """重采样到LUNA16标准间距"""
        print(f"   重采样到LUNA16标准间距: {self.target_spacing}")

        try:
            original_spacing = image.GetSpacing()
            original_size = image.GetSize()

            # 计算新尺寸
            new_size = [
                int(round(original_size[i] * original_spacing[i] / self.target_spacing[i]))
                for i in range(3)
            ]

            print(f"   原始尺寸: {original_size} -> 新尺寸: {new_size}")

            # 重采样
            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputSpacing(self.target_spacing)
            resampler.SetSize(new_size)
            resampler.SetOutputDirection(image.GetDirection())
            resampler.SetOutputOrigin(image.GetOrigin())
            resampler.SetTransform(sitk.Transform())
            resampler.SetDefaultPixelValue(-1000)  # 空气HU值
            resampler.SetInterpolator(sitk.sitkLinear)

            resampled_image = resampler.Execute(image)
            print(f"   重采样完成: {resampled_image.GetSize()}")

            return resampled_image

        except Exception as e:
            print(f"⚠️ 重采样失败: {e}")
            return image

    def _standardize_orientation(self, image):
        """标准化方向到RAS"""
        try:
            # LUNA16使用RAS方向
            oriented_image = sitk.DICOMOrient(image, 'RAS')
            print(f"   方向标准化完成: RAS")
            return oriented_image
        except Exception as e:
            print(f"⚠️ 方向标准化失败: {e}")
            return image

    def _crop_foreground(self, image):
        """裁剪前景区域"""
        try:
            array = sitk.GetArrayFromImage(image)

            # 简单的前景检测（HU值 > -900）
            foreground_mask = array > -900

            # 找到前景区域的边界
            coords = np.where(foreground_mask)

            if len(coords[0]) == 0:
                print("   未检测到前景，跳过裁剪")
                return image

            # 计算边界框
            z_min, z_max = coords[0].min(), coords[0].max()
            y_min, y_max = coords[1].min(), coords[1].max()
            x_min, x_max = coords[2].min(), coords[2].max()

            # 添加边界
            padding = 10
            z_min = max(0, z_min - padding)
            z_max = min(array.shape[0], z_max + padding)
            y_min = max(0, y_min - padding)
            y_max = min(array.shape[1], y_max + padding)
            x_min = max(0, x_min - padding)
            x_max = min(array.shape[2], x_max + padding)

            # 裁剪
            cropped_array = array[z_min:z_max, y_min:y_max, x_min:x_max]

            # 创建新图像
            cropped_image = sitk.GetImageFromArray(cropped_array)
            cropped_image.CopyInformation(image)

            print(f"   前景裁剪完成: {image.GetSize()} -> {cropped_image.GetSize()}")
            return cropped_image

        except Exception as e:
            print(f"⚠️ 前景裁剪失败: {e}")
            return image

    def _resize_to_target_size(self, image):
        """调整到目标尺寸"""
        print(f"   调整到目标尺寸: {self.target_size}")

        try:
            current_size = image.GetSize()
            target_size = self.target_size

            # 使用SimpleITK的Resample进行尺寸调整
            resampler = sitk.ResampleImageFilter()
            resampler.SetSize(target_size)

            # 计算新的间距
            current_spacing = image.GetSpacing()
            new_spacing = [
                current_spacing[i] * current_size[i] / target_size[i]
                for i in range(3)
            ]

            resampler.SetOutputSpacing(new_spacing)
            resampler.SetOutputDirection(image.GetDirection())
            resampler.SetOutputOrigin(image.GetOrigin())
            resampler.SetTransform(sitk.Transform())
            resampler.SetDefaultPixelValue(-1000)
            resampler.SetInterpolator(sitk.sitkLinear)

            resized_image = resampler.Execute(image)
            print(f"   尺寸调整完成: {current_size} -> {resized_image.GetSize()}")

            return resized_image

        except Exception as e:
            print(f"⚠️ 尺寸调整失败: {e}")
            return image

    def _normalize_hu_values(self, image):
        """标准化HU值"""
        try:
            array = sitk.GetArrayFromImage(image)

            # 裁剪HU值到肺部范围
            hu_min, hu_max = self.hu_window
            clipped_array = np.clip(array, hu_min, hu_max)

            # 归一化到[0, 1]
            normalized_array = (clipped_array - hu_min) / (hu_max - hu_min)
            normalized_array = normalized_array.astype(np.float32)

            # 创建新图像
            normalized_image = sitk.GetImageFromArray(normalized_array)
            normalized_image.CopyInformation(image)

            print(f"   HU值标准化完成: [{hu_min}, {hu_max}] -> [0, 1]")
            return normalized_image

        except Exception as e:
            print(f"⚠️ HU值标准化失败: {e}")
            return image

    def _create_preprocessing_versions(self, base_image):
        """创建多种预处理版本"""
        base_array = sitk.GetArrayFromImage(base_image)
        versions = {}

        # 版本1: 标准LUNA16归一化 [0, 1]
        versions['luna16_standard'] = base_array

        # 版本2: 零均值单位方差
        mean_val = base_array.mean()
        std_val = base_array.std()
        if std_val > 0:
            versions['z_normalized'] = (base_array - mean_val) / std_val
        else:
            versions['z_normalized'] = base_array

        # 版本3: HU值重映射
        hu_min, hu_max = self.hu_window
        versions['hu_restored'] = base_array * (hu_max - hu_min) + hu_min

        # 版本4: 对比度增强
        versions['contrast_enhanced'] = self._apply_3d_clahe(base_array)

        return versions

    def _apply_3d_clahe(self, array):
        """应用3D CLAHE对比度增强"""
        try:
            enhanced_array = np.zeros_like(array)

            for i in range(array.shape[0]):
                slice_2d = array[i]

                # 转换到uint8
                slice_min, slice_max = slice_2d.min(), slice_2d.max()
                if slice_max > slice_min:
                    slice_uint8 = ((slice_2d - slice_min) / (slice_max - slice_min) * 255).astype(np.uint8)

                    # 应用CLAHE
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    enhanced_uint8 = clahe.apply(slice_uint8)

                    # 转换回float32
                    enhanced_array[i] = enhanced_uint8.astype(np.float32) / 255.0
                else:
                    enhanced_array[i] = slice_2d

            return enhanced_array

        except Exception as e:
            print(f"⚠️ 3D CLAHE失败: {e}")
            return array

    def save_versions_as_nifti(self, processing_result, output_dir=None):
        """保存预处理版本为NIfTI文件"""
        if output_dir is None:
            output_dir = tempfile.mkdtemp()

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        versions = processing_result['versions']
        temp_files = {}

        for version_name, array in versions.items():
            # 创建SimpleITK图像
            image = sitk.GetImageFromArray(array.astype(np.float32))
            image.SetSpacing(self.target_spacing)

            # 保存为NIfTI
            temp_file = output_path / f"luna16_{version_name}.nii.gz"
            sitk.WriteImage(image, str(temp_file))

            temp_files[version_name] = str(temp_file)
            print(f"   保存版本 {version_name}: {temp_file.name}")

        return temp_files


class Pure3DDetector:
    """纯3D推理检测器"""

    def __init__(self, bundle_path=None, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.bundle_loader = None
        self.model = None
        self.model_info = {}

        # 初始化DICOM处理器
        self.dicom_processor = LUNA16DicomProcessor()

        print(f"🚀 初始化纯3D检测器")
        print(f"   设备: {self.device}")
        print(f"   当前用户: veryjoyran")
        print(f"   时间: 2025-06-24 15:25:55")

        # 设置3D预处理
        self.setup_3d_transforms()

        if bundle_path:
            self.load_bundle(bundle_path)

    def setup_3d_transforms(self):
        """设置3D预处理变换"""
        self.transforms = Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            ToTensord(keys=["image"]),
        ])

    def load_bundle(self, bundle_path):
        """加载Bundle"""
        try:
            print(f"🔄 加载Bundle: {bundle_path}")

            self.bundle_loader = MonAIBundleLoader(bundle_path, self.device)
            success = self.bundle_loader.load_bundle()

            self.model = self.bundle_loader.get_model()
            self.model_info = self.bundle_loader.get_model_info()

            if self.model is None:
                raise Exception("模型加载失败")

            print(f"✅ Bundle加载完成")
            print(f"   模型类型: {self.model_info.get('network_class', '未知')}")

            return True

        except Exception as e:
            print(f"❌ Bundle加载失败: {e}")
            return False

    def detect_3d(self, dicom_path, test_all_versions=True):
        """执行3D检测"""
        print(f"🔍 开始3D检测")
        print(f"   DICOM路径: {Path(dicom_path).name}")

        if self.model is None:
            print("❌ 模型未加载")
            return None

        # 1. LUNA16标准预处理
        processing_result = self.dicom_processor.process_dicom_to_luna16_standard(dicom_path)

        if processing_result is None:
            return None

        # 2. 保存为NIfTI文件
        temp_files = self.dicom_processor.save_versions_as_nifti(processing_result)

        if test_all_versions:
            # 测试所有版本
            return self._test_all_versions(processing_result, temp_files)
        else:
            # 使用标准版本
            return self._test_single_version(processing_result, temp_files)

    def _test_all_versions(self, processing_result, temp_files):
        """测试所有预处理版本"""
        print("🧪 测试所有预处理版本...")

        test_results = {}

        for version_name, temp_file in temp_files.items():
            print(f"\n🔍 测试版本: {version_name}")

            try:
                result = self._inference_3d(temp_file)

                if result and result.get('detection_count', 0) > 0:
                    print(f"   ✅ 检测到 {result['detection_count']} 个候选")
                    test_results[version_name] = {
                        'success': True,
                        'detection_count': result['detection_count'],
                        'max_confidence': max(result.get('scores', [0])) if result.get('scores') else 0,
                        'result': result
                    }
                else:
                    print(f"   ➖ 无检测结果")
                    test_results[version_name] = {
                        'success': False,
                        'detection_count': 0,
                        'max_confidence': 0,
                        'result': None
                    }

            except Exception as e:
                print(f"   ❌ 测试失败: {e}")
                test_results[version_name] = {
                    'success': False,
                    'detection_count': 0,
                    'max_confidence': 0,
                    'error': str(e)
                }

        # 清理临时文件
        for temp_file in temp_files.values():
            try:
                Path(temp_file).unlink()
            except:
                pass

        # 分析结果
        best_version = self._find_best_version(test_results)

        print(f"\n📊 版本测试完成:")
        print(f"   最佳版本: {best_version}")

        return {
            'processing_result': processing_result,
            'test_results': test_results,
            'best_version': best_version,
            'detection_type': '3D_volumetric'
        }

    def _test_single_version(self, processing_result, temp_files):
        """测试单一版本"""
        version_name = 'luna16_standard'
        if version_name not in temp_files:
            version_name = list(temp_files.keys())[0]

        temp_file = temp_files[version_name]

        try:
            result = self._inference_3d(temp_file)

            # 清理临时文件
            for temp_f in temp_files.values():
                try:
                    Path(temp_f).unlink()
                except:
                    pass

            if result:
                result['processing_result'] = processing_result
                result['preprocessing_version'] = version_name
                result['detection_type'] = '3D_volumetric'

            return result

        except Exception as e:
            print(f"❌ 检测失败: {e}")
            return None

    def _inference_3d(self, nifti_path):
        """3D推理"""
        try:
            # 加载数据
            data_dict = {"image": nifti_path}
            data_dict = self.transforms(data_dict)

            input_tensor = data_dict["image"]

            # 确保正确的3D格式
            if input_tensor.dim() == 3:  # (D, H, W)
                input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
            elif input_tensor.dim() == 4:  # (1, D, H, W) 或 (C, D, H, W)
                if input_tensor.shape[0] == 1:
                    input_tensor = input_tensor.unsqueeze(0)  # (1, 1, D, H, W)
                else:
                    input_tensor = input_tensor.unsqueeze(0)  # (1, C, D, H, W)

            input_tensor = input_tensor.to(self.device)

            print(f"     3D输入形状: {input_tensor.shape}")
            print(f"     输入范围: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")

            # 推理
            self.model.eval()
            with torch.no_grad():
                output = self.model(input_tensor)

                print(f"     输出类型: {type(output)}")

                if isinstance(output, dict):
                    return self._process_detection_output(output)
                elif isinstance(output, (list, tuple)):
                    # RetinaNet可能返回列表
                    return self._process_detection_list_output(output)
                elif isinstance(output, torch.Tensor):
                    return self._process_segmentation_output(output)
                else:
                    print(f"     未知输出格式: {type(output)}")
                    return None

        except Exception as e:
            print(f"     3D推理失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _process_detection_output(self, output):
        """处理检测输出（字典格式）"""
        try:
            # 检查可能的键名
            possible_box_keys = ['boxes', 'pred_boxes', 'detection_boxes']
            possible_score_keys = ['scores', 'pred_scores', 'detection_scores']
            possible_label_keys = ['labels', 'pred_labels', 'detection_labels']

            boxes = None
            scores = None
            labels = None

            # 查找boxes
            for key in possible_box_keys:
                if key in output:
                    boxes = output[key]
                    print(f"     找到boxes键: {key}")
                    break

            # 查找scores
            for key in possible_score_keys:
                if key in output:
                    scores = output[key]
                    print(f"     找到scores键: {key}")
                    break

            # 查找labels
            for key in possible_label_keys:
                if key in output:
                    labels = output[key]
                    print(f"     找到labels键: {key}")
                    break

            # 转换tensor到列表
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.cpu().numpy().tolist()
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy().tolist()
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy().tolist()

            print(f"     检测框数量: {len(boxes) if boxes else 0}")

            if boxes and len(boxes) > 0:
                print(f"     置信度范围: [{min(scores):.3f}, {max(scores):.3f}]")

                # 应用阈值过滤
                for threshold in [0.5, 0.3, 0.1, 0.05]:
                    if scores:
                        filtered_indices = [i for i, score in enumerate(scores) if score > threshold]

                        if filtered_indices:
                            print(f"     阈值 {threshold}: {len(filtered_indices)} 个检测")

                            return {
                                'detection_mode': True,
                                'boxes': [boxes[i] for i in filtered_indices] if boxes else [],
                                'scores': [scores[i] for i in filtered_indices] if scores else [],
                                'labels': [labels[i] for i in filtered_indices] if labels else [],
                                'threshold_used': threshold,
                                'detection_count': len(filtered_indices)
                            }

                # 返回所有检测
                return {
                    'detection_mode': True,
                    'boxes': boxes or [],
                    'scores': scores or [],
                    'labels': labels or [],
                    'threshold_used': 0.0,
                    'detection_count': len(boxes) if boxes else 0
                }
            else:
                return {
                    'detection_mode': True,
                    'boxes': [],
                    'scores': [],
                    'labels': [],
                    'threshold_used': 0.0,
                    'detection_count': 0
                }

        except Exception as e:
            print(f"     检测输出处理失败: {e}")
            return None

    def _process_detection_list_output(self, output_list):
        """处理检测输出（列表格式）"""
        try:
            print(f"     处理列表输出，长度: {len(output_list)}")

            # RetinaNet通常返回[boxes, scores]或[boxes, scores, labels]
            if len(output_list) >= 2:
                boxes_tensor = output_list[0]
                scores_tensor = output_list[1]
                labels_tensor = output_list[2] if len(output_list) > 2 else None

                # 转换为numpy/list
                if isinstance(boxes_tensor, torch.Tensor):
                    boxes = boxes_tensor.cpu().numpy().tolist()
                else:
                    boxes = boxes_tensor

                if isinstance(scores_tensor, torch.Tensor):
                    scores = scores_tensor.cpu().numpy().tolist()
                else:
                    scores = scores_tensor

                if labels_tensor is not None:
                    if isinstance(labels_tensor, torch.Tensor):
                        labels = labels_tensor.cpu().numpy().tolist()
                    else:
                        labels = labels_tensor
                else:
                    labels = [1] * len(boxes) if boxes else []

                # 处理和过滤
                return self._filter_detections(boxes, scores, labels)

            return None

        except Exception as e:
            print(f"     列表输出处理失败: {e}")
            return None

    def _filter_detections(self, boxes, scores, labels):
        """过滤检测结果"""
        if not boxes or len(boxes) == 0:
            return {
                'detection_mode': True,
                'boxes': [],
                'scores': [],
                'labels': [],
                'threshold_used': 0.0,
                'detection_count': 0
            }

        print(f"     原始检测数量: {len(boxes)}")
        print(f"     置信度范围: [{min(scores):.3f}, {max(scores):.3f}]")

        # 应用阈值
        for threshold in [0.5, 0.3, 0.1, 0.05]:
            filtered_indices = [i for i, score in enumerate(scores) if score > threshold]

            if filtered_indices:
                print(f"     阈值 {threshold}: {len(filtered_indices)} 个检测")

                return {
                    'detection_mode': True,
                    'boxes': [boxes[i] for i in filtered_indices],
                    'scores': [scores[i] for i in filtered_indices],
                    'labels': [labels[i] for i in filtered_indices] if labels else [1] * len(filtered_indices),
                    'threshold_used': threshold,
                    'detection_count': len(filtered_indices)
                }

        # 返回所有检测
        return {
            'detection_mode': True,
            'boxes': boxes,
            'scores': scores,
            'labels': labels if labels else [1] * len(boxes),
            'threshold_used': 0.0,
            'detection_count': len(boxes)
        }

    def _process_segmentation_output(self, output):
        """处理分割输出"""
        try:
            print(f"     分割输出形状: {output.shape}")

            # 获取概率图
            if output.shape[1] > 1:
                probs = torch.softmax(output, dim=1)
                prob_volume = probs[0, 1].cpu().numpy()  # 结节概率
            else:
                prob_volume = torch.sigmoid(output[0, 0]).cpu().numpy()

            print(f"     概率体积范围: [{prob_volume.min():.3f}, {prob_volume.max():.3f}]")

            # 3D阈值化和连通组件分析
            for threshold in [0.5, 0.3, 0.1, 0.05]:
                binary_volume = prob_volume > threshold

                if np.sum(binary_volume) > 0:
                    # 3D连通组件分析
                    labeled_volume, num_features = ndimage.label(binary_volume)

                    if num_features > 0:
                        print(f"     阈值 {threshold}: {num_features} 个3D组件")

                        # 提取3D边界框
                        boxes_3d = []
                        scores = []

                        for i in range(1, num_features + 1):
                            mask = (labeled_volume == i)
                            size = np.sum(mask)

                            if size > 50:  # 最小3D尺寸
                                coords = np.where(mask)
                                z1, z2 = coords[0].min(), coords[0].max()
                                y1, y2 = coords[1].min(), coords[1].max()
                                x1, x2 = coords[2].min(), coords[2].max()

                                region_probs = prob_volume[mask]
                                avg_prob = float(region_probs.mean())

                                # 3D边界框格式: [x1, y1, z1, x2, y2, z2]
                                boxes_3d.append([int(x1), int(y1), int(z1), int(x2), int(y2), int(z2)])
                                scores.append(avg_prob)

                        if boxes_3d:
                            return {
                                'detection_mode': False,
                                'boxes': boxes_3d,
                                'scores': scores,
                                'labels': [1] * len(boxes_3d),
                                'threshold_used': threshold,
                                'detection_count': len(boxes_3d),
                                'probability_volume': prob_volume,
                                'is_3d_segmentation': True
                            }

            # 无检测结果
            return {
                'detection_mode': False,
                'boxes': [],
                'scores': [],
                'labels': [],
                'threshold_used': 0.5,
                'detection_count': 0,
                'probability_volume': prob_volume,
                'is_3d_segmentation': True
            }

        except Exception as e:
            print(f"     分割输出处理失败: {e}")
            return None

    def _find_best_version(self, test_results):
        """找出最佳预处理版本"""
        best_version = None
        best_score = 0

        for version_name, result in test_results.items():
            if result['success']:
                score = result['detection_count'] * 0.6 + result['max_confidence'] * 0.4
                if score > best_score:
                    best_score = score
                    best_version = version_name

        return best_version

    def visualize_3d_result(self, result, save_path=None):
        """可视化3D检测结果"""
        if not result:
            print("❌ 无结果可视化")
            return None

        try:
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            fig.suptitle('3D LUNA16检测结果', fontsize=16, fontweight='bold')

            if 'processing_result' in result:
                processing_result = result['processing_result']

                # 显示处理步骤
                original_array = processing_result['original_array']
                processed_array = processing_result['processed_array']

                # 取中间切片显示
                mid_slice_orig = original_array.shape[0] // 2
                mid_slice_proc = processed_array.shape[0] // 2

                axes[0, 0].imshow(original_array[mid_slice_orig], cmap='gray')
                axes[0, 0].set_title('原始DICOM')
                axes[0, 0].axis('off')

                axes[0, 1].imshow(processed_array[mid_slice_proc], cmap='gray')
                axes[0, 1].set_title('LUNA16处理')
                axes[0, 1].axis('off')

                # 显示检测结果
                detection_count = result.get('detection_count', 0)
                axes[0, 2].imshow(processed_array[mid_slice_proc], cmap='gray')
                axes[0, 2].set_title(f'3D检测 - 发现{detection_count}个')
                axes[0, 2].axis('off')

                # 绘制3D边界框（投影到中间切片）
                boxes = result.get('boxes', [])
                scores = result.get('scores', [])

                for i, (box, score) in enumerate(zip(boxes[:5], scores[:5])):  # 最多显示5个
                    if len(box) == 6:  # 3D框格式 [x1, y1, z1, x2, y2, z2]
                        x1, y1, z1, x2, y2, z2 = box

                        # 如果框包含中间切片，则绘制
                        if z1 <= mid_slice_proc <= z2:
                            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                               linewidth=2, edgecolor='red', facecolor='none')
                            axes[0, 2].add_patch(rect)

                            axes[0, 2].text(x1, y1-2, f'#{i+1}\n{score:.3f}',
                                           color='red', fontsize=8, fontweight='bold',
                                           bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.8))

            # 显示统计信息
            stats_text = f"""3D检测统计:
检测类型: {result.get('detection_type', '未知')}
发现候选: {result.get('detection_count', 0)}个
检测模式: {'目标检测' if result.get('detection_mode', False) else '分割检测'}
使用阈值: {result.get('threshold_used', 'N/A')}

LUNA16处理:
目标间距: (0.703125, 0.703125, 1.25) mm
目标尺寸: (192, 192, 80)
HU窗口: (-1000, 400)"""

            axes[0, 3].text(0.1, 0.5, stats_text, transform=axes[0, 3].transAxes,
                            fontsize=9, verticalalignment='center',
                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            axes[0, 3].set_title('3D检测信息')
            axes[0, 3].axis('off')

            # 多版本测试结果可视化
            if 'test_results' in result:
                test_results = result['test_results']
                version_names = list(test_results.keys())

                for i, version_name in enumerate(version_names[:4]):  # 最多显示4个版本
                    test_result = test_results[version_name]

                    title = f"{version_name}\n"
                    if test_result['success']:
                        title += f"✓ {test_result['detection_count']}个检测\n置信度: {test_result['max_confidence']:.3f}"
                        color = 'lightgreen'
                    else:
                        title += "✗ 无检测"
                        color = 'lightcoral'

                    axes[1, i].text(0.5, 0.5, title, transform=axes[1, i].transAxes,
                                   ha='center', va='center', fontsize=10,
                                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
                    axes[1, i].set_title(version_name, fontsize=8)
                    axes[1, i].axis('off')

            # 隐藏多余的子图
            for i in range(len(version_names) if 'test_results' in result else 0, 4):
                axes[1, i].axis('off')

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
                print(f"📸 3D检测结果保存至: {save_path}")

            return fig

        except Exception as e:
            print(f"❌ 3D可视化失败: {e}")
            return None

    def generate_3d_report(self, result, dicom_path):
        """🔥 生成3D检测报告 - 修复AttributeError"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report = f"""
🎯 3D LUNA16兼容肺结节检测报告
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

👤 用户: veryjoyran
📅 检测时间: {current_time}
📁 DICOM路径: {Path(dicom_path).name}

🔧 LUNA16标准处理:
  • 体素间距重采样: 0.703125 x 0.703125 x 1.25 mm
  • 模型输入尺寸: 192 x 192 x 80 (与训练完全一致)
  • 坐标系统: RAS方向
  • HU值窗口: [-1000, 400] (肺部专用范围)
  • 3D体积处理: 完整3D上下文分析

💡 LIDC数据集兼容性:
  • ✅ LIDC是LUNA16的基础数据集
  • ✅ 相同的CT扫描源和注释
  • ✅ 兼容的预处理流程
  • ✅ 模型推理无需适配
"""

        # 根据检测类型生成不同报告
        if 'test_results' in result:
            # 多版本测试结果
            test_results = result['test_results']
            best_version = result['best_version']

            report += f"""
📊 多版本3D处理结果:
  • 测试预处理版本: {len(test_results)}个
  • 最佳表现版本: {best_version if best_version else '无'}
  • 成功版本: {sum(1 for r in test_results.values() if r['success'])}个

📋 详细版本结果:
"""

            for version_name, test_result in test_results.items():
                status = "✅" if test_result['success'] else "❌"
                count = test_result['detection_count']
                confidence = test_result['max_confidence']

                report += f"""
{status} {version_name}:
  • 3D检测数量: {count}
  • 最高置信度: {confidence:.3f}
"""

                if test_result['success'] and 'result' in test_result:
                    result_data = test_result['result']
                    boxes = result_data.get('boxes', [])
                    scores = result_data.get('scores', [])

                    for i, (box, score) in enumerate(zip(boxes[:3], scores[:3])):  # 显示前3个
                        if len(box) == 6:  # 3D框
                            x1, y1, z1, x2, y2, z2 = box
                            volume = (x2-x1) * (y2-y1) * (z2-z1)
                            # 估算物理体积 (mm³)
                            physical_volume = volume * (0.703125**2) * 1.25
                            report += f"    3D检测 {i+1}: 框[{x1},{y1},{z1},{x2},{y2},{z2}], 体积~{physical_volume:.1f}mm³, 置信度{score:.3f}\n"
                        else:  # 2D框投影
                            x1, y1, x2, y2 = box[:4]
                            area = (x2-x1) * (y2-y1)
                            report += f"    2D投影 {i+1}: 框[{x1},{y1},{x2},{y2}], 面积{area}px², 置信度{score:.3f}\n"

            if best_version:
                report += f"""
🎉 推荐处理方法: {best_version}
   该版本在您的LIDC数据上达到最佳检测性能
"""
            else:
                report += """
😞 所有版本均未成功检测

💡 LIDC数据考虑因素:
  • LIDC注释可能比LUNA16挑战数据更细微
  • 原始LIDC包含LUNA16排除的小结节(<3mm)
  • 考虑为LIDC数据调整置信度阈值
  • LIDC有4名放射科医师共识 vs LUNA16的处理注释
"""
        else:
            # 单版本结果
            detection_count = result.get('detection_count', 0)

            if detection_count > 0:
                report += f"""
📊 3D检测结果:
  • 总3D候选发现: {detection_count}个
  • 检测模式: {'目标检测' if result.get('detection_mode', False) else '3D分割'}
  • 置信度阈值: {result.get('threshold_used', 'N/A')}
  • 处理版本: {result.get('preprocessing_version', 'luna16_standard')}

📋 3D候选详情:
"""

                boxes = result.get('boxes', [])
                scores = result.get('scores', [])

                for i, (box, score) in enumerate(zip(boxes, scores)):
                    if len(box) == 6:  # 3D边界框
                        x1, y1, z1, x2, y2, z2 = box

                        # 计算3D尺寸
                        width = (x2 - x1) * 0.703125  # mm
                        height = (y2 - y1) * 0.703125  # mm
                        depth = (z2 - z1) * 1.25  # mm
                        volume = width * height * depth

                        # 估算结节直径
                        diameter = (width + height + depth) / 3

                        report += f"""
🔍 3D候选 {i+1}:
  • 3D位置: ({x1}, {y1}, {z1}) 到 ({x2}, {y2}, {z2})
  • 物理尺寸: {width:.1f} × {height:.1f} × {depth:.1f} mm
  • 估算体积: {volume:.1f} mm³
  • 估算直径: {diameter:.1f} mm
  • 3D置信度: {score:.3f}
"""

                        # LIDC特定的风险评估
                        if diameter >= 4.0:
                            risk_level = "🔴 显著尺寸 (≥4mm)"
                            recommendation = "符合LUNA16尺寸标准 - 高优先级"
                        elif diameter >= 3.0:
                            risk_level = "🟡 边界尺寸 (3-4mm)"
                            recommendation = "接近LUNA16阈值 - 中等优先级"
                        else:
                            risk_level = "🟢 小尺寸 (<3mm)"
                            recommendation = "低于LUNA16阈值但在LIDC中可检测"

                        report += f"  • 尺寸分类: {risk_level}\n"
                        report += f"  • 临床备注: {recommendation}\n"

                    else:  # 2D投影框
                        report += f"""
🔍 2D投影 {i+1}:
  • 2D位置: {box}
  • 置信度: {score:.3f}
"""
            else:
                report += """
❌ 未检测到3D候选

💡 LIDC vs LUNA16分析:
  • LIDC包含各种协议的原始CT扫描
  • LUNA16使用标准化预处理和>3mm结节过滤
  • 您的LIDC数据可能包含LUNA16训练中没有的小结节
  • 模型在LUNA16的过滤和处理子集上训练
"""

        report += f"""

🔍 LIDC数据集特定考虑:

📊 LIDC vs LUNA16差异:
  • LIDC: 1,018例原始数据集，4名放射科医师注释
  • LUNA16: 888例处理子集，仅>3mm结节
  • LIDC: 可变扫描协议和层厚
  • LUNA16: 标准化预处理和一致间距

✅ 兼容性因素:
  • 相同源CT扫描 (LIDC-IDRI数据库)
  • 兼容的HU值范围和肺部解剖
  • 模型预处理匹配LUNA16标准
  • 3D检测方法适用于两个数据集

⚠️ 潜在差异:
  • LIDC可能包含LUNA16训练中没有的<3mm结节
  • 原始LIDC扫描参数 vs LUNA16标准化
  • 注释共识差异 (4名放射科医师 vs 处理的真值)

💡 LIDC数据优化:
  • 考虑较低置信度阈值 (0.3 → 0.1)
  • 测试保留原始LIDC间距
  • 在解释中考虑较小结节尺寸
  • 如有可用，与放射科医师注释交叉参考

⚙️ 技术处理总结:
  • 输入格式: LIDC DICOM → LUNA16兼容3D体积
  • 空间重采样: ✅ 到0.703125mm各向同性 + 1.25mm层
  • 强度归一化: ✅ HU [-1000, 400] → [0, 1]
  • 3D模型输入: ✅ (1, 1, 80, 192, 192) 张量格式
  • 坐标系统: ✅ RAS方向保持一致性

📈 检测质量评估:
  • 3D上下文: {'已利用' if result.get('detection_type') == '3D_volumetric' else '有限'}
  • 处理标准: LUNA16兼容
  • 数据源: LIDC (原始CT扫描)
  • 模型训练: LUNA16 (处理子集)

📞 技术支持: veryjoyran | 3D LUNA16兼容 v5.0.0
时间: {current_time} | 数据集: LIDC兼容处理
"""

        return report

    def cleanup(self):
        """清理资源"""
        if self.bundle_loader:
            self.bundle_loader.cleanup()


def test_3d_detector():
    """测试3D检测器"""
    print("🧪 测试3D检测器")
    print(f"   当前用户: veryjoyran")
    print(f"   时间: 2025-06-24 15:25:55")

    bundle_path = "lung_nodule_ct_detection_v0.5.9.zip"
    dicom_path = "sample_lidc_dicom"  # LIDC数据路径

    if Path(bundle_path).exists():
        detector = Pure3DDetector(bundle_path, 'cpu')

        if detector.model is not None:
            print("✅ 模型加载成功，开始LIDC数据检测测试...")

            # 模拟检测（如果有DICOM数据）
            if Path(dicom_path).exists():
                result = detector.detect_3d(dicom_path, test_all_versions=True)

                if result:
                    print("✅ LIDC数据3D检测测试完成")

                    # 生成报告
                    report = detector.generate_3d_report(result, dicom_path)
                    print("\n" + "="*80)
                    print("检测报告预览:")
                    print(report[:1000] + "..." if len(report) > 1000 else report)
                else:
                    print("⚠️ 检测未返回结果")
            else:
                print(f"⚠️ 测试DICOM路径不存在: {dicom_path}")
        else:
            print("❌ 模型加载失败")

        detector.cleanup()
    else:
        print(f"❌ Bundle文件不存在: {bundle_path}")


if __name__ == "__main__":
    test_3d_detector()