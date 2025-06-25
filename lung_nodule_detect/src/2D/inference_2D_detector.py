"""
2D肺结节推理检测器
Author: veryjoyran
Date: 2025-06-24 03:04:13
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import SimpleITK as sitk
import tempfile
import json
from datetime import datetime

# MonAI 2D推理相关导入
import monai
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    Orientationd, ScaleIntensityRanged, ToTensord,
    ResizeWithPadOrCropd, SpatialPadd
)
from scipy import ndimage
from skimage import measure, morphology

# 导入2D Bundle加载器
from bundle_loader_2D import Bundle2DLoader

print(f"🔧 MonAI版本: {monai.__version__}")
print(f"🔧 PyTorch版本: {torch.__version__}")
print(f"🔧 设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")


class Inference2DDetector:
    """2D肺结节推理检测器"""

    def __init__(self, bundle_path=None, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.bundle_loader = None
        self.model = None
        self.model_info = {}
        self.is_detection_model = False

        print(f"🚀 初始化2D推理检测器")
        print(f"   设备: {self.device}")

        # 设置2D预处理变换
        self.setup_2d_inference_transforms()

        # 如果提供了bundle路径，立即加载
        if bundle_path:
            self.load_bundle_2d(bundle_path)

    def setup_2d_inference_transforms(self):
        """设置2D推理专用的数据预处理"""
        print("🔧 设置2D推理预处理流水线...")

        self.inference_transforms_2d = Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-1000,
                a_max=400,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            # 2D标准化尺寸
            ResizeWithPadOrCropd(
                keys=["image"],
                spatial_size=(512, 512),
                mode="constant",
                constant_values=0
            ),
            ToTensord(keys=["image"]),
        ])

        print("✅ 2D推理预处理流水线设置完成")

    def load_bundle_2d(self, bundle_path):
        """加载MonAI Bundle - 2D模式"""
        try:
            print(f"🔄 加载Bundle (2D模式): {bundle_path}")

            self.bundle_loader = Bundle2DLoader(bundle_path, self.device)
            success = self.bundle_loader.load_bundle_for_2d()

            self.model = self.bundle_loader.get_model()
            self.model_info = self.bundle_loader.get_model_info()

            if self.model is None:
                raise Exception("2D模型加载失败")

            print(f"✅ 2D Bundle加载完成")
            print(f"   模型类型: {self.model_info.get('type', 'Unknown')}")
            print(f"   原始类型: {self.model_info.get('original_type', 'Unknown')}")
            print(f"   预训练: {self.model_info.get('pretrained', False)}")
            print(f"   权重加载比例: {self.model_info.get('loaded_ratio', 0):.2f}")
            print(f"   参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"   2D适配: {self.model_info.get('adapted_to_2d', False)}")

            # 检查是否是检测模型
            if 'RetinaNet' in self.model_info.get('type', '') or 'RetinaNet' in self.model_info.get('original_type',
                                                                                                    ''):
                print("🎯 检测到RetinaNet模型，将使用2D目标检测推理模式")
                self.is_detection_model = True
            else:
                print("🎯 检测到分割模型，将使用2D分割推理模式")
                self.is_detection_model = False

            return True

        except Exception as e:
            print(f"❌ 2D Bundle加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def convert_dicom_to_2d_slices(self, dicom_path):
        """将DICOM转换为2D切片"""
        print(f"🔄 转换DICOM为2D切片: {dicom_path}")

        try:
            dicom_path = Path(dicom_path)

            if dicom_path.is_dir():
                # DICOM序列目录
                reader = sitk.ImageSeriesReader()
                dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_path))

                if not dicom_names:
                    raise ValueError("目录中未找到DICOM文件")

                reader.SetFileNames(dicom_names)
                image = reader.Execute()
                print(f"   读取DICOM序列: {len(dicom_names)} 个文件")

            elif dicom_path.suffix.lower() == '.dcm':
                # 单个DICOM文件
                image = sitk.ReadImage(str(dicom_path))
                print(f"   读取单个DICOM文件")
            else:
                raise ValueError(f"不支持的文件类型: {dicom_path.suffix}")

            # 获取基本信息
            print(f"   原始尺寸: {image.GetSize()}")
            print(f"   体素间距: {[f'{x:.2f}' for x in image.GetSpacing()]}")

            # 标准化方向
            image = sitk.DICOMOrient(image, 'LPS')

            # 获取3D数组
            array_3d = sitk.GetArrayFromImage(image)
            print(f"   HU值范围: [{array_3d.min():.1f}, {array_3d.max():.1f}]")
            print(f"   3D数组形状: {array_3d.shape}")

            # 提取所有2D切片
            slices_info = []

            for i in range(array_3d.shape[0]):
                slice_2d = array_3d[i]

                # 保存为临时NIfTI文件
                temp_file = tempfile.mktemp(suffix=f'_slice_{i:03d}.nii.gz')

                # 创建2D SimpleITK图像
                image_2d = sitk.GetImageFromArray(slice_2d)
                if len(image.GetSpacing()) >= 2:
                    image_2d.SetSpacing(image.GetSpacing()[:2])

                sitk.WriteImage(image_2d, temp_file)

                slice_info = {
                    'slice_index': i,
                    'array': slice_2d,
                    'temp_file': temp_file,
                    'image_2d': image_2d,
                    'hu_min': float(slice_2d.min()),
                    'hu_max': float(slice_2d.max()),
                    'hu_mean': float(slice_2d.mean()),
                    'shape': slice_2d.shape
                }

                slices_info.append(slice_info)

            print(f"✅ 转换完成: {len(slices_info)} 个2D切片")

            return slices_info, image

        except Exception as e:
            print(f"❌ DICOM转换失败: {e}")
            return None, None

    def inference_2d_single_slice(self, nifti_path_2d):
        """对单个2D切片进行推理"""
        if self.model is None:
            print("❌ 模型未加载")
            return None

        print(f"🔍 开始2D单切片推理")
        print(f"   输入: {Path(nifti_path_2d).name}")

        try:
            # 准备数据
            data_dict = {"image": nifti_path_2d}

            # 应用预处理
            data_dict = self.inference_transforms_2d(data_dict)

            # 获取处理后的张量
            input_tensor = data_dict["image"]
            print(f"   预处理后形状: {input_tensor.shape}")

            # 确保有批次维度
            if input_tensor.dim() == 2:  # (H, W)
                input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            elif input_tensor.dim() == 3:  # (C, H, W)
                input_tensor = input_tensor.unsqueeze(0)  # (1, C, H, W)

            input_tensor = input_tensor.to(self.device)
            print(f"   最终输入形状: {input_tensor.shape}")

            # 🔥 根据模型类型选择推理方法
            if self.is_detection_model:
                return self._detection_inference_2d(input_tensor)
            else:
                return self._segmentation_inference_2d(input_tensor)

        except Exception as e:
            print(f"❌ 2D推理失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _detection_inference_2d(self, input_tensor):
        """2D目标检测推理"""
        print(f"   执行2D目标检测推理...")

        try:
            self.model.eval()
            with torch.no_grad():

                print(f"   🔍 2D检测详细信息:")
                print(f"     输入张量尺寸: {input_tensor.shape}")
                print(f"     输入数据类型: {input_tensor.dtype}")
                print(f"     输入范围: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")

                # 直接调用模型
                output = self.model(input_tensor)

                print(f"     输出类型: {type(output)}")

                if isinstance(output, dict):
                    print("     输出是字典格式:")
                    for key, value in output.items():
                        if isinstance(value, torch.Tensor):
                            print(f"       {key}: 形状={value.shape}")
                            if value.numel() > 0:
                                print(f"             范围=[{value.min():.3f}, {value.max():.3f}]")
                        elif isinstance(value, list):
                            print(f"       {key}: 列表长度={len(value)}")
                        else:
                            print(f"       {key}: {type(value)}")

                    boxes = output.get('boxes', [])
                    scores = output.get('scores', [])
                    labels = output.get('labels', [])

                    print(f"     检测结果:")
                    print(f"       检测框数量: {len(boxes) if hasattr(boxes, '__len__') else 0}")

                    if hasattr(boxes, '__len__') and len(boxes) > 0:
                        print(f"       🎯 发现检测框!")

                        # 应用多个阈值
                        thresholds = [0.5, 0.3, 0.1, 0.05]

                        for threshold in thresholds:
                            high_conf_indices = [i for i, score in enumerate(scores) if score > threshold]

                            if high_conf_indices:
                                filtered_boxes = [boxes[i] for i in high_conf_indices]
                                filtered_scores = [scores[i] for i in high_conf_indices]
                                filtered_labels = [labels[i] for i in high_conf_indices] if labels else []

                                print(f"       阈值 {threshold}: {len(filtered_boxes)} 个检测")

                                if threshold <= 0.3:  # 使用较低的阈值
                                    return {
                                        'boxes': filtered_boxes,
                                        'scores': filtered_scores,
                                        'labels': filtered_labels,
                                        'input_shape': input_tensor.shape,
                                        'detection_count': len(filtered_boxes),
                                        'detection_mode': True,
                                        'threshold_used': threshold
                                    }

                        # 如果所有阈值都没有结果，使用极低阈值
                        print(f"       使用所有检测结果 (无阈值过滤)")
                        return {
                            'boxes': boxes,
                            'scores': scores,
                            'labels': labels,
                            'input_shape': input_tensor.shape,
                            'detection_count': len(boxes),
                            'detection_mode': True,
                            'threshold_used': 0.0
                        }

                    else:
                        print(f"       ❌ 未检测到任何目标")
                        return {
                            'boxes': [],
                            'scores': [],
                            'labels': [],
                            'input_shape': input_tensor.shape,
                            'detection_count': 0,
                            'detection_mode': True,
                            'threshold_used': 0.0
                        }

                # 处理其他输出格式
                elif isinstance(output, torch.Tensor):
                    print(f"     直接张量输出，尝试作为分割结果处理")
                    return self._process_segmentation_output_2d(output, input_tensor)

                else:
                    print(f"     未知输出格式: {type(output)}")
                    return None

        except Exception as e:
            print(f"⚠️ 2D目标检测推理失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _segmentation_inference_2d(self, input_tensor):
        """2D语义分割推理"""
        print(f"   执行2D语义分割推理...")

        try:
            self.model.eval()
            with torch.no_grad():

                # 直接推理
                output = self.model(input_tensor)

                print(f"   原始推理输出形状: {output.shape}")

                return self._process_segmentation_output_2d(output, input_tensor)

        except Exception as e:
            print(f"⚠️ 2D分割推理失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _process_segmentation_output_2d(self, output, input_tensor):
        """处理2D分割输出"""
        try:
            # 应用softmax获取概率
            if output.shape[1] > 1:  # 多类输出
                probs = torch.softmax(output, dim=1)
                prob_map = probs[0, 1].cpu().numpy()  # 结节概率
            else:
                prob_map = torch.sigmoid(output[0, 0]).cpu().numpy()

            print(f"   概率图统计: min={prob_map.min():.3f}, max={prob_map.max():.3f}, mean={prob_map.mean():.3f}")

            # 应用阈值
            threshold = 0.5
            binary_mask = prob_map > threshold

            print(f"   阈值({threshold})后阳性像素: {np.sum(binary_mask)}")

            # 形态学处理
            if np.sum(binary_mask) > 0:
                # 2D形态学开运算
                kernel = morphology.disk(2)
                binary_mask = morphology.binary_opening(binary_mask, kernel)
                print(f"   形态学开运算后阳性像素: {np.sum(binary_mask)}")

                # 2D形态学闭运算
                kernel = morphology.disk(1)
                binary_mask = morphology.binary_closing(binary_mask, kernel)
                print(f"   形态学闭运算后阳性像素: {np.sum(binary_mask)}")

            # 连通组件分析
            labeled_array, num_features = ndimage.label(binary_mask)
            print(f"   连通组件数量: {num_features}")

            # 提取边界框
            boxes = []
            scores = []
            labels = []

            for i in range(1, num_features + 1):
                mask = (labeled_array == i)
                size = np.sum(mask)

                if size > 10:  # 最小尺寸
                    # 计算边界框
                    coords = np.where(mask)
                    y1, y2 = coords[0].min(), coords[0].max()
                    x1, x2 = coords[1].min(), coords[1].max()

                    # 计算平均概率作为置信度
                    region_probs = prob_map[mask]
                    avg_prob = float(region_probs.mean())

                    if avg_prob > 0.3:  # 概率阈值
                        boxes.append([int(x1), int(y1), int(x2), int(y2)])
                        scores.append(avg_prob)
                        labels.append(1)

            print(f"✅ 2D分割推理完成，提取到 {len(boxes)} 个候选区域")

            return {
                'boxes': boxes,
                'scores': scores,
                'labels': labels,
                'input_shape': input_tensor.shape,
                'detection_count': len(boxes),
                'detection_mode': False,  # 分割模式
                'probability_map': prob_map,
                'binary_mask': binary_mask
            }

        except Exception as e:
            print(f"❌ 2D分割输出处理失败: {e}")
            return None

    def batch_inference_all_slices(self, dicom_path, confidence_threshold=0.3, max_slices=None):
        """批量推理所有2D切片"""
        print(f"🔍 批量2D推理所有切片")

        # 转换为2D切片
        slices_info, original_image = self.convert_dicom_to_2d_slices(dicom_path)

        if not slices_info:
            return None

        # 限制处理的切片数量（用于测试）
        if max_slices:
            slices_info = slices_info[:max_slices]
            print(f"   限制处理切片数量: {max_slices}")

        all_detections = []
        total_slices = len(slices_info)

        print(f"   开始逐切片推理 ({total_slices} 个切片)...")

        for i, slice_info in enumerate(slices_info):
            print(f"   处理切片 {i + 1}/{total_slices} (索引: {slice_info['slice_index']})")

            # 推理单个切片
            inference_result = self.inference_2d_single_slice(slice_info['temp_file'])

            if inference_result and inference_result['detection_count'] > 0:
                # 过滤低置信度检测
                boxes = inference_result['boxes']
                scores = inference_result['scores']
                labels = inference_result['labels']

                high_conf_indices = [j for j, score in enumerate(scores) if score > confidence_threshold]

                if high_conf_indices:
                    filtered_result = {
                        'slice_index': slice_info['slice_index'],
                        'slice_array': slice_info['array'],
                        'boxes': [boxes[j] for j in high_conf_indices],
                        'scores': [scores[j] for j in high_conf_indices],
                        'labels': [labels[j] for j in high_conf_indices] if labels else [],
                        'detection_count': len(high_conf_indices),
                        'detection_mode': inference_result.get('detection_mode', False),
                        'threshold_used': inference_result.get('threshold_used', confidence_threshold),
                        'hu_stats': {
                            'min': slice_info['hu_min'],
                            'max': slice_info['hu_max'],
                            'mean': slice_info['hu_mean']
                        }
                    }

                    # 如果是分割模式，保存概率图
                    if 'probability_map' in inference_result:
                        filtered_result['probability_map'] = inference_result['probability_map']
                    if 'binary_mask' in inference_result:
                        filtered_result['binary_mask'] = inference_result['binary_mask']

                    all_detections.append(filtered_result)

                    print(f"     ✅ 切片 {i + 1} 检测到 {len(high_conf_indices)} 个高置信度候选")
                else:
                    print(f"     ➖ 切片 {i + 1} 有检测但置信度过低")
            else:
                print(f"     ➖ 切片 {i + 1} 无检测结果")

        # 清理临时文件
        for slice_info in slices_info:
            try:
                Path(slice_info['temp_file']).unlink()
            except:
                pass

        print(f"✅ 批量2D推理完成")
        print(f"   有检测结果的切片: {len(all_detections)}/{total_slices}")

        if all_detections:
            total_detections = sum(d['detection_count'] for d in all_detections)
            max_confidence = max(max(d['scores']) for d in all_detections)
            avg_confidence = np.mean([score for d in all_detections for score in d['scores']])

            print(f"   总检测数量: {total_detections}")
            print(f"   最高置信度: {max_confidence:.3f}")
            print(f"   平均置信度: {avg_confidence:.3f}")

        return all_detections

    def extract_candidates_2d(self, detection_results, min_confidence=0.3):
        """从2D检测结果中提取候选结节"""
        print(f"🎯 提取2D候选结节")
        print(f"   最小置信度: {min_confidence}")

        if not detection_results:
            print("   无检测结果")
            return []

        candidates = []
        candidate_id = 1

        for slice_result in detection_results:
            slice_idx = slice_result['slice_index']
            boxes = slice_result['boxes']
            scores = slice_result['scores']

            for box, score in zip(boxes, scores):
                if score >= min_confidence:
                    x1, y1, x2, y2 = box

                    candidate = {
                        "id": candidate_id,
                        "slice_index": slice_idx,
                        "bbox_2d": [x1, y1, x2, y2],
                        "center_2d": [(x1 + x2) / 2, (y1 + y2) / 2],
                        "size_2d": (x2 - x1) * (y2 - y1),
                        "confidence": score,
                        "detection_mode": slice_result.get('detection_mode', False)
                    }

                    candidates.append(candidate)
                    candidate_id += 1

        # 按置信度排序
        candidates = sorted(candidates, key=lambda x: x["confidence"], reverse=True)

        print(f"✅ 提取到 {len(candidates)} 个2D候选结节")

        # 显示前几个候选
        for i, cand in enumerate(candidates[:5]):
            print(
                f"   候选 {i + 1}: 切片#{cand['slice_index']}, 置信度={cand['confidence']:.3f}, 尺寸={cand['size_2d']:.0f}")

        return candidates


# 2D可视化器
class Simple2DVisualizer:
    """简单的2D结果可视化器"""

    @staticmethod
    def create_2d_detection_overview(detection_results, save_path=None):
        """创建2D检测结果总览"""

        if not detection_results:
            print("❌ 无检测结果可视化")
            return None

        # 选择最佳的检测结果进行展示
        sorted_results = sorted(detection_results,
                                key=lambda x: max(x['scores']) if x['scores'] else 0,
                                reverse=True)

        # 最多显示8个最佳切片
        top_results = sorted_results[:8]

        cols = 4
        rows = 2
        fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
        fig.suptitle(f'2D肺结节检测总览 - {len(detection_results)}个切片有检测结果',
                     fontsize=16, fontweight='bold')

        for i, result in enumerate(top_results):
            if i >= rows * cols:
                break

            row = i // cols
            col = i % cols
            ax = axes[row, col]

            slice_array = result['slice_array']
            slice_idx = result['slice_index']

            # 显示切片
            ax.imshow(slice_array, cmap='gray', vmin=slice_array.min(), vmax=slice_array.max())
            ax.set_title(f'切片 #{slice_idx} - {result["detection_count"]} 个检测')
            ax.axis('off')

            # 绘制检测框
            boxes = result['boxes']
            scores = result['scores']

            for j, (box, score) in enumerate(zip(boxes, scores)):
                x1, y1, x2, y2 = box

                # 绘制边界框
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)

                # 添加标签
                ax.text(x1, y1 - 3, f'#{j + 1}\n{score:.2f}',
                        color='red', fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.8))

        # 隐藏多余的子图
        for i in range(len(top_results), rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"📸 2D检测总览保存至: {save_path}")

        return fig

    @staticmethod
    def create_2d_candidates_montage(candidates, detection_results, save_path=None):
        """创建2D候选结节蒙太奇"""

        if not candidates:
            print("❌ 无2D候选结节可视化")
            return None

        # 为候选结节找到对应的切片数据
        slice_data_map = {r['slice_index']: r['slice_array'] for r in detection_results}

        n_candidates = min(len(candidates), 12)  # 最多显示12个候选

        cols = 4
        rows = (n_candidates + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(f'2D候选结节详细视图 (前{n_candidates}个)', fontsize=16, fontweight='bold')

        for i in range(n_candidates):
            row = i // cols
            col = i % cols
            ax = axes[row, col]

            cand = candidates[i]
            slice_idx = cand['slice_index']

            if slice_idx in slice_data_map:
                slice_array = slice_data_map[slice_idx]
                x1, y1, x2, y2 = cand['bbox_2d']

                # 提取候选区域的patch
                patch_size = 64
                center_x, center_y = cand['center_2d']

                patch_x1 = max(0, int(center_x - patch_size // 2))
                patch_x2 = min(slice_array.shape[1], int(center_x + patch_size // 2))
                patch_y1 = max(0, int(center_y - patch_size // 2))
                patch_y2 = min(slice_array.shape[0], int(center_y + patch_size // 2))

                patch = slice_array[patch_y1:patch_y2, patch_x1:patch_x2]

                # 显示patch
                ax.imshow(patch, cmap='gray')
                ax.set_title(f'候选#{cand["id"]} (切片#{slice_idx})\n置信度:{cand["confidence"]:.3f}', fontsize=10)
                ax.axis('off')

                # 在patch中心画十字
                h, w = patch.shape
                ax.plot([w // 2 - 5, w // 2 + 5], [h // 2, h // 2], 'r-', linewidth=2)
                ax.plot([w // 2, w // 2], [h // 2 - 5, h // 2 + 5], 'r-', linewidth=2)
            else:
                ax.text(0.5, 0.5, f'候选#{cand["id"]}\n切片#{slice_idx}\n数据缺失',
                        transform=ax.transAxes, ha='center', va='center')
                ax.axis('off')

        # 隐藏多余的子图
        for i in range(n_candidates, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"📸 2D候选蒙太奇保存至: {save_path}")

        return fig


def save_2d_inference_results(candidates, detection_results, model_info, output_dir="./2d_inference_results"):
    """保存2D推理结果"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_data = {
        "user": "veryjoyran",
        "timestamp": timestamp,
        "inference_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": "2d_inference_v1.0.0",
        "mode": "2d_slice_by_slice",
        "model_info": model_info,
        "summary": {
            "total_slices_with_detections": len(detection_results),
            "total_candidates": len(candidates),
            "total_detections": sum(d['detection_count'] for d in detection_results),
            "max_confidence": max([max(d['scores']) for d in detection_results]) if detection_results else 0,
            "avg_confidence": np.mean(
                [score for d in detection_results for score in d['scores']]) if detection_results else 0
        },
        "candidates": candidates,
        "detection_results": [
            {
                "slice_index": d["slice_index"],
                "detection_count": d["detection_count"],
                "boxes": d["boxes"],
                "scores": d["scores"],
                "detection_mode": d.get("detection_mode", False),
                "hu_stats": d.get("hu_stats", {})
            }
            for d in detection_results
        ]
    }

    json_path = output_path / f"2d_inference_results_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False, default=str)

    print(f"💾 2D推理结果保存至: {json_path}")
    return json_path


def run_2d_inference(bundle_path, dicom_path, output_dir="./2d_inference_results"):
    """运行完整的2D推理流程"""
    print("🚀 开始2D肺结节检测推理")
    print(f"👤 用户: veryjoyran")
    print(f"📅 时间: 2025-01-24 03:10:12")
    print("=" * 60)

    try:
        # 1. 初始化2D检测器
        print("\n📋 第1步: 初始化2D检测器")
        detector = Inference2DDetector(bundle_path)

        if detector.model is None:
            print("❌ 2D检测器初始化失败")
            return None

        # 2. 批量推理所有切片
        print("\n📋 第2步: 批量推理所有2D切片")
        detection_results = detector.batch_inference_all_slices(
            dicom_path,
            confidence_threshold=0.3,
            max_slices=None  # 处理所有切片
        )

        if not detection_results:
            print("❌ 所有切片都无检测结果")
            return None

        # 3. 提取候选结节
        print("\n📋 第3步: 提取2D候选结节")
        candidates = detector.extract_candidates_2d(detection_results, min_confidence=0.3)

        if not candidates:
            print("❌ 未提取到有效候选结节")
            return None

        # 4. 生成可视化
        print("\n📋 第4步: 生成2D可视化")
        visualizer = Simple2DVisualizer()

        # 创建检测总览
        overview_fig = visualizer.create_2d_detection_overview(
            detection_results,
            save_path=Path(output_dir) / "2d_detection_overview.png"
        )

        # 创建候选蒙太奇
        montage_fig = visualizer.create_2d_candidates_montage(
            candidates,
            detection_results,
            save_path=Path(output_dir) / "2d_candidates_montage.png"
        )

        # 5. 保存结果
        print("\n📋 第5步: 保存2D推理结果")
        json_path = save_2d_inference_results(
            candidates,
            detection_results,
            detector.model_info,
            output_dir
        )

        print(f"\n🎉 2D推理完成!")
        print(f"   有检测结果的切片: {len(detection_results)}")
        print(f"   候选结节数量: {len(candidates)}")
        print(f"   最高置信度: {max(c['confidence'] for c in candidates):.3f}")
        print(f"   结果保存目录: {output_dir}")

        return {
            "detector": detector,
            "detection_results": detection_results,
            "candidates": candidates,
            "json_path": json_path,
            "overview_fig": overview_fig,
            "montage_fig": montage_fig
        }

    except Exception as e:
        print(f"❌ 2D推理失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # 使用示例
    bundle_path = "lung_nodule_ct_detection_v0.5.9.zip"
    dicom_path = "path/to/your/dicom/data"

    if Path(bundle_path).exists() and Path(dicom_path).exists():
        result = run_2d_inference(bundle_path, dicom_path)

        if result:
            print("✅ 2D推理成功完成!")
        else:
            print("❌ 2D推理失败")
    else:
        print("❌ 请检查Bundle和DICOM路径")