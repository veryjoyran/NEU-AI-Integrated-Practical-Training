import cv2
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import matplotlib.pyplot as plt
import tempfile


class ImprovedDicomProcessor:
    """改进的DICOM处理器 - 基于YOLO成功经验"""

    def __init__(self):
        print("🚀 初始化改进版DICOM处理器")

    def load_and_preprocess_dicom(self, dicom_path, target_size=(512, 512),
                                  window_center=50, window_width=350):
        """
        加载并预处理DICOM文件
        复用YOLO中成功的预处理逻辑
        """
        print(f"🔄 加载DICOM文件: {dicom_path}")

        try:
            # 1. 读取DICOM文件
            image = sitk.ReadImage(str(dicom_path))
            array = sitk.GetArrayFromImage(image)

            print(f"   原始形状: {array.shape}")
            print(f"   原始HU范围: [{array.min():.1f}, {array.max():.1f}]")

            # 2. 如果是3D，选择最佳切片
            if array.ndim == 3:
                best_slice_idx = self._select_best_slice(array)
                array_2d = array[best_slice_idx]
                print(f"   选择切片 #{best_slice_idx}")
            else:
                array_2d = array

            # 3. 应用窗宽窗位 (复用YOLO的窗口设置)
            windowed_array = self._apply_window_level(array_2d, window_center, window_width)
            print(f"   窗宽窗位处理: C={window_center}, W={window_width}")

            # 4. 归一化到[0, 255] (YOLO使用的范围)
            normalized_array = self._normalize_to_uint8(windowed_array)

            # 5. 对比度增强 (YOLO中的CLAHE)
            enhanced_array = self._apply_clahe(normalized_array)

            # 6. 调整尺寸
            resized_array = cv2.resize(enhanced_array, target_size, interpolation=cv2.INTER_CUBIC)

            print(f"   最终处理形状: {resized_array.shape}")
            print(f"   最终值范围: [{resized_array.min()}, {resized_array.max()}]")

            # 7. 创建多种预处理版本用于测试
            versions = self._create_preprocessing_versions(resized_array)

            return {
                'original_array': array_2d,
                'windowed_array': windowed_array,
                'enhanced_array': enhanced_array,
                'final_array': resized_array,
                'versions': versions,
                'preprocessing_info': {
                    'window_center': window_center,
                    'window_width': window_width,
                    'target_size': target_size,
                    'original_shape': array.shape,
                    'original_hu_range': [float(array.min()), float(array.max())]
                }
            }

        except Exception as e:
            print(f"❌ DICOM预处理失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _select_best_slice(self, array_3d):
        """
        选择最佳切片 - 复用YOLO的切片选择逻辑
        选择标准差最大的切片（通常包含更多结构信息）
        """
        slice_scores = []

        for i in range(array_3d.shape[0]):
            slice_2d = array_3d[i]

            # 计算切片质量指标
            std_score = np.std(slice_2d)  # 标准差
            mean_score = np.mean(slice_2d)  # 平均值
            range_score = np.max(slice_2d) - np.min(slice_2d)  # 动态范围

            # 组合评分 (优先选择有更多细节的切片)
            combined_score = std_score * 0.5 + range_score * 0.3 + abs(mean_score + 500) * 0.2
            slice_scores.append(combined_score)

        best_idx = np.argmax(slice_scores)
        print(f"   切片质量评分最高: #{best_idx} (评分: {slice_scores[best_idx]:.2f})")

        return best_idx

    def _apply_window_level(self, array, center, width):
        """
        应用窗宽窗位 - 复用YOLO的窗口设置
        """
        min_val = center - width // 2
        max_val = center + width // 2

        # 裁剪到窗口范围
        windowed = np.clip(array, min_val, max_val)

        return windowed

    def _normalize_to_uint8(self, array):
        """
        归一化到[0, 255] - YOLO使用的数据范围
        """
        min_val = array.min()
        max_val = array.max()

        if max_val > min_val:
            normalized = ((array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(array, dtype=np.uint8)

        return normalized

    def _apply_clahe(self, array):
        """
        应用CLAHE对比度增强 - 复用YOLO的对比度增强
        """
        # 创建CLAHE对象
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # 应用CLAHE
        enhanced = clahe.apply(array)

        return enhanced

    def _create_preprocessing_versions(self, base_array):
        """
        创建多种预处理版本用于测试
        """
        versions = {}

        # 版本1: 原始归一化到[0, 1] (MonAI标准)
        versions['monai_normalized'] = (base_array.astype(np.float32) / 255.0)

        # 版本2: 标准化 (零均值单位方差)
        mean_val = base_array.mean()
        std_val = base_array.std()
        if std_val > 0:
            versions['standardized'] = ((base_array.astype(np.float32) - mean_val) / std_val)
        else:
            versions['standardized'] = base_array.astype(np.float32)

        # 版本3: HU值重映射 (医学图像标准)
        versions['hu_remapped'] = self._remap_to_hu_range(base_array)

        # 版本4: 直接uint8
        versions['uint8_direct'] = base_array

        return versions

    def _remap_to_hu_range(self, array):
        """
        重新映射到医学标准HU值范围
        """
        # 将[0, 255]映射回典型的肺部HU范围[-1000, 400]
        hu_min, hu_max = -1000, 400
        remapped = (array.astype(np.float32) / 255.0) * (hu_max - hu_min) + hu_min

        return remapped

    def visualize_preprocessing_steps(self, processing_result, save_path=None):
        """
        可视化预处理步骤
        """
        if processing_result is None:
            return None

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('DICOM预处理步骤可视化', fontsize=16, fontweight='bold')

        # 原始图像
        axes[0, 0].imshow(processing_result['original_array'], cmap='gray')
        axes[0, 0].set_title('1. 原始DICOM')
        axes[0, 0].axis('off')

        # 窗宽窗位处理
        axes[0, 1].imshow(processing_result['windowed_array'], cmap='gray')
        axes[0, 1].set_title('2. 窗宽窗位处理')
        axes[0, 1].axis('off')

        # 对比度增强
        axes[0, 2].imshow(processing_result['enhanced_array'], cmap='gray')
        axes[0, 2].set_title('3. CLAHE对比度增强')
        axes[0, 2].axis('off')

        # 最终处理结果
        axes[1, 0].imshow(processing_result['final_array'], cmap='gray')
        axes[1, 0].set_title('4. 最终处理结果')
        axes[1, 0].axis('off')

        # 直方图对比
        axes[1, 1].hist(processing_result['original_array'].flatten(), bins=50, alpha=0.7, label='原始', color='blue')
        axes[1, 1].hist(processing_result['final_array'].flatten(), bins=50, alpha=0.7, label='处理后', color='red')
        axes[1, 1].set_title('5. 像素值分布对比')
        axes[1, 1].legend()

        # 处理信息
        info = processing_result['preprocessing_info']
        info_text = f"""预处理参数:
窗宽: {info['window_width']}
窗位: {info['window_center']}
目标尺寸: {info['target_size']}
原始形状: {info['original_shape']}
原始HU范围: [{info['original_hu_range'][0]:.1f}, {info['original_hu_range'][1]:.1f}]"""

        axes[1, 2].text(0.1, 0.5, info_text, transform=axes[1, 2].transAxes,
                        fontsize=10, verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 2].set_title('6. 处理信息')
        axes[1, 2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"📸 预处理可视化保存至: {save_path}")

        return fig

    def save_preprocessing_versions_as_nifti(self, processing_result, output_dir="./preprocessing_versions"):
        """
        保存不同预处理版本为NIfTI文件，用于测试
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        versions = processing_result['versions']
        temp_files = {}

        for version_name, array in versions.items():
            # 确保数组有正确的维度
            if array.ndim == 2:
                array_3d = array[np.newaxis, ...]  # 添加第三维
            else:
                array_3d = array

            # 创建SimpleITK图像
            image = sitk.GetImageFromArray(array_3d.astype(np.float32))

            # 保存为临时NIfTI文件
            temp_file = output_path / f"version_{version_name}.nii.gz"
            sitk.WriteImage(image, str(temp_file))

            temp_files[version_name] = str(temp_file)

            print(f"   保存版本 {version_name}: {temp_file.name}")

        return temp_files


class MultiVersionDetector:
    """多版本预处理测试检测器"""

    def __init__(self, detector, dicom_processor):
        self.detector = detector
        self.dicom_processor = dicom_processor

    def test_all_preprocessing_versions(self, dicom_path):
        """
        测试所有预处理版本，找出最有效的方法
        """
        print("🧪 开始多版本预处理测试")

        # 1. 预处理DICOM
        processing_result = self.dicom_processor.load_and_preprocess_dicom(dicom_path)

        if processing_result is None:
            print("❌ DICOM预处理失败")
            return None

        # 2. 保存预处理版本为临时文件
        temp_files = self.dicom_processor.save_preprocessing_versions_as_nifti(processing_result)

        # 3. 测试每个版本
        test_results = {}

        for version_name, temp_file in temp_files.items():
            print(f"\n🔍 测试版本: {version_name}")

            try:
                # 使用检测器测试这个版本
                result = self.detector.detect_single_dicom_from_nifti(temp_file)

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

        # 4. 清理临时文件
        for temp_file in temp_files.values():
            try:
                Path(temp_file).unlink()
            except:
                pass

        # 5. 分析结果
        self._analyze_test_results(test_results)

        return {
            'processing_result': processing_result,
            'test_results': test_results,
            'best_version': self._find_best_version(test_results)
        }

    def _analyze_test_results(self, test_results):
        """分析测试结果"""
        print("\n📊 多版本测试结果分析:")
        print("=" * 50)

        successful_versions = []

        for version_name, result in test_results.items():
            status = "✅" if result['success'] else "❌"
            detection_count = result['detection_count']
            max_confidence = result['max_confidence']

            print(f"{status} {version_name:15} | 检测数: {detection_count:2d} | 最高置信度: {max_confidence:.3f}")

            if result['success']:
                successful_versions.append(version_name)

        print("=" * 50)

        if successful_versions:
            print(f"🎉 成功的预处理版本: {', '.join(successful_versions)}")
        else:
            print("😞 所有版本都未检测到结节")
            print("💡 建议:")
            print("   • 尝试不同的窗宽窗位设置")
            print("   • 检查DICOM文件是否包含肺部结构")
            print("   • 验证Bundle模型是否正确")
            print("   • 考虑使用其他切片")

    def _find_best_version(self, test_results):
        """找出最佳的预处理版本"""
        best_version = None
        best_score = 0

        for version_name, result in test_results.items():
            if result['success']:
                # 综合评分：检测数量 + 最高置信度
                score = result['detection_count'] * 0.6 + result['max_confidence'] * 0.4

                if score > best_score:
                    best_score = score
                    best_version = version_name

        return best_version