import numpy as np
import cv2
from scipy import ndimage
from skimage import morphology, measure, segmentation
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import warnings


class LungSegmentationPreprocessor:
    """
    改进的肺部分割预处理器
    修复字体显示问题和警告
    """

    def __init__(self):
        self.debug_mode = False
        self.save_debug_images = False
        self.debug_output_dir = None
        self.show_comparison = False
        self._setup_matplotlib()

    def _setup_matplotlib(self):
        """设置matplotlib，解决中文显示和警告问题"""
        try:
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
            plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负数显示问题

            # 禁用字体相关警告
            warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

            if self.debug_mode:
                print("  字体设置完成")

        except Exception as e:
            if self.debug_mode:
                print(f"  字体设置失败: {e}")
            # 如果中文字体设置失败，使用英文
            plt.rcParams['font.family'] = ['DejaVu Sans']

    def enable_debug(self, save_images=True, output_dir="debug_lung_seg", show_comparison=True):
        """启用调试模式"""
        self.debug_mode = True
        self.save_debug_images = save_images
        self.show_comparison = show_comparison
        self._setup_matplotlib()  # 重新设置matplotlib

        if save_images:
            self.debug_output_dir = Path(output_dir)
            self.debug_output_dir.mkdir(exist_ok=True)

    def _debug_save(self, image, filename, title=""):
        """保存调试图像（使用英文标题避免字体问题）"""
        if self.save_debug_images and self.debug_output_dir:
            plt.figure(figsize=(10, 8))
            plt.imshow(image, cmap='gray')

            # 使用英文标题避免字体警告
            english_title = self._translate_title(title)
            plt.title(english_title, fontsize=12)
            plt.axis('off')
            plt.tight_layout()

            save_path = self.debug_output_dir / f"{filename}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            if self.debug_mode:
                print(f"  Debug: Saved {save_path}")

    def _translate_title(self, chinese_title):
        """将中文标题翻译为英文，避免字体警告"""
        translation_dict = {
            "二值化": "Binary Segmentation",
            "去除小区域": "Remove Small Objects",
            "填充空洞": "Fill Holes",
            "肺实质提取": "Lung Parenchyma Extraction",
            "清理后肺掩码": "Cleaned Lung Mask",
            "左肺掩码": "Left Lung Mask",
            "右肺掩码": "Right Lung Mask",
            "最终肺掩码": "Final Lung Mask",
            "最终分割结果": "Final Segmentation Result",
            "精细化结果": "Refinement Result",
            "原图": "Original Image",
            "分割结果": "Segmentation Result",
            "轮廓叠加": "Contour Overlay",
            "肺分割": "Lung Segmentation"
        }

        # 处理包含参数的标题
        for chinese, english in translation_dict.items():
            if chinese in chinese_title:
                # 保留参数部分
                if "=" in chinese_title or "(" in chinese_title:
                    # 提取参数部分
                    import re
                    params = re.findall(r'\([^)]*\)', chinese_title)
                    param_str = ' '.join(params) if params else ""

                    # 提取数值参数
                    numbers = re.findall(r'=(\d+)', chinese_title)
                    if numbers:
                        param_str += f" (threshold={numbers[0]})" if "阈值" in chinese_title else f" (size={numbers[0]})"

                    return f"{english} {param_str}"
                else:
                    return english

        # 如果没有找到对应翻译，返回简化的英文标题
        return chinese_title.replace("阈值", "threshold").replace("面积", "area")

    def _show_comparison(self, original, segmented, title_prefix="", filename_debug=""):
        """显示原图和分割结果的对比（使用英文标题）"""
        if self.show_comparison:
            plt.figure(figsize=(15, 6))

            # 转换标题为英文
            english_prefix = self._translate_title(title_prefix)

            plt.subplot(1, 3, 1)
            plt.imshow(original, cmap='gray')
            plt.title(f'{english_prefix} - Original')
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(segmented, cmap='gray')
            plt.title(f'{english_prefix} - Segmented')
            plt.axis('off')

            # 第三个子图显示掩码轮廓叠加
            plt.subplot(1, 3, 3)
            plt.imshow(original, cmap='gray')

            # 找到肺区域边界并叠加显示
            if isinstance(segmented, np.ndarray) and segmented.dtype == bool:
                # 如果是掩码，找边界
                boundaries = segmentation.find_boundaries(segmented, mode='thick')
                boundary_coords = np.where(boundaries)
                plt.scatter(boundary_coords[1], boundary_coords[0], c='red', s=0.1, alpha=0.7)
            else:
                # 如果是分割后的图像，创建掩码
                mask = segmented > 0
                boundaries = segmentation.find_boundaries(mask, mode='thick')
                boundary_coords = np.where(boundaries)
                plt.scatter(boundary_coords[1], boundary_coords[0], c='red', s=0.1, alpha=0.7)

            plt.title(f'{english_prefix} - Overlay')
            plt.axis('off')

            plt.tight_layout()

            if self.save_debug_images and self.debug_output_dir:
                save_path = self.debug_output_dir / f"{filename_debug}_comparison.png"
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                if self.debug_mode:
                    print(f"  Comparison: Saved comparison plot {save_path}")

            plt.show()

    def segment_lungs_8bit(self, image_8bit, filename_debug=""):
        """
        基于MATLAB算法的8位图像肺分割

        Args:
            image_8bit: 8位灰度图像 (uint8 array, 0-255)
            filename_debug: 用于调试的文件名

        Returns:
            tuple: (lung_mask, processed_image, lung_bbox, left_lung_mask, right_lung_mask)
        """
        try:
            if self.debug_mode:
                print(f"    🫁 8-bit Lung Segmentation: {filename_debug}")
                print(f"      Input pixel range: [{image_8bit.min()}, {image_8bit.max()}]")
                print(f"      Image size: {image_8bit.shape}")

            original_image = image_8bit.copy()

            # 步骤1: 自动阈值分割 (对应MATLAB的graythresh和im2bw)
            threshold_val = threshold_otsu(image_8bit)
            binary_mask = image_8bit > threshold_val

            if self.debug_mode:
                print(f"      Otsu threshold: {threshold_val:.1f}")

            self._debug_save((binary_mask * 255).astype(np.uint8),
                             f"{filename_debug}_01_binary_mask", f"Binary (thresh={threshold_val:.1f})")

            # 步骤2: 去除小区域 (对应MATLAB的bwareaopen)
            min_area = max(1000, int(image_8bit.size * 0.005))
            cleaned_mask = morphology.remove_small_objects(binary_mask, min_size=min_area)

            self._debug_save((cleaned_mask * 255).astype(np.uint8),
                             f"{filename_debug}_02_cleaned", f"Remove Small Objects (min_area={min_area})")

            # 步骤3: 填充空洞 (对应MATLAB的imfill)
            filled_mask = ndimage.binary_fill_holes(cleaned_mask)

            self._debug_save((filled_mask * 255).astype(np.uint8),
                             f"{filename_debug}_03_filled", "Fill Holes")

            # 步骤4: 提取肺实质 (对应MATLAB的D-C操作)
            lung_parenchyma = filled_mask & (~cleaned_mask)

            self._debug_save((lung_parenchyma * 255).astype(np.uint8),
                             f"{filename_debug}_04_parenchyma", "Lung Parenchyma")

            # 步骤5: 再次填充肺实质空洞
            lung_filled = ndimage.binary_fill_holes(lung_parenchyma)

            # 步骤6: 去除小区域，保留主要肺区域
            final_min_area = max(2000, int(image_8bit.size * 0.01))
            lung_mask_clean = morphology.remove_small_objects(lung_filled, min_size=final_min_area)

            self._debug_save((lung_mask_clean * 255).astype(np.uint8),
                             f"{filename_debug}_05_lung_clean", f"Clean Lung Mask (min_area={final_min_area})")

            # 步骤7: 分离左右肺
            labeled_lungs = measure.label(lung_mask_clean)
            regions = measure.regionprops(labeled_lungs)

            if self.debug_mode:
                print(f"      Found {len(regions)} connected components")

            if len(regions) == 0:
                if self.debug_mode:
                    print("      ⚠️  No lung regions found, using full image")
                return np.ones_like(image_8bit, dtype=bool), image_8bit, None, None, None

            # 按面积排序，取最大的两个区域作为左右肺
            regions_sorted = sorted(regions, key=lambda x: x.area, reverse=True)

            left_lung_mask = np.zeros_like(labeled_lungs, dtype=bool)
            right_lung_mask = np.zeros_like(labeled_lungs, dtype=bool)

            # 第一个最大区域
            if len(regions_sorted) >= 1:
                left_lung_mask = (labeled_lungs == regions_sorted[0].label)

            # 第二个最大区域
            if len(regions_sorted) >= 2:
                right_lung_mask = (labeled_lungs == regions_sorted[1].label)

            # 根据质心位置确定左右肺
            if len(regions_sorted) >= 2:
                centroid1 = regions_sorted[0].centroid
                centroid2 = regions_sorted[1].centroid

                # 如果第一个区域的质心在右侧，则交换
                if centroid1[1] > centroid2[1]:
                    left_lung_mask, right_lung_mask = right_lung_mask, left_lung_mask

            self._debug_save((left_lung_mask * 255).astype(np.uint8),
                             f"{filename_debug}_06_left_lung", "Left Lung Mask")
            self._debug_save((right_lung_mask * 255).astype(np.uint8),
                             f"{filename_debug}_06_right_lung", "Right Lung Mask")

            # 步骤8: 左右肺分别进行形态学修复
            processed_left = self._refine_lung_mask(left_lung_mask, "left", filename_debug)
            processed_right = self._refine_lung_mask(right_lung_mask, "right", filename_debug)

            # 步骤9: 合并左右肺得到最终掩码
            final_lung_mask = processed_left | processed_right

            self._debug_save((final_lung_mask * 255).astype(np.uint8),
                             f"{filename_debug}_07_final_mask", "Final Lung Mask")

            # 步骤10: 应用掩码到原图像
            processed_image = original_image.copy()
            processed_image[~final_lung_mask] = 0  # 肺外区域设为黑色

            self._debug_save(processed_image, f"{filename_debug}_08_result", "Final Result")

            # 步骤11: 计算肺区域边界框
            lung_coords = np.where(final_lung_mask)
            if len(lung_coords[0]) > 0:
                min_row, max_row = lung_coords[0].min(), lung_coords[0].max()
                min_col, max_col = lung_coords[1].min(), lung_coords[1].max()
                lung_bbox = (min_row, min_col, max_row, max_col)
            else:
                lung_bbox = None

            # 显示对比图
            self._show_comparison(original_image, processed_image, "Lung Segmentation", filename_debug)

            if self.debug_mode:
                lung_area = np.sum(final_lung_mask)
                total_area = final_lung_mask.size
                lung_percentage = (lung_area / total_area) * 100
                print(f"      ✅ Lung segmentation completed:")
                print(f"        Lung area ratio: {lung_percentage:.1f}%")
                print(f"        Left lung area: {np.sum(processed_left)}")
                print(f"        Right lung area: {np.sum(processed_right)}")
                if lung_bbox:
                    print(f"        Lung bbox: ({lung_bbox[0]},{lung_bbox[1]}) to ({lung_bbox[2]},{lung_bbox[3]})")

            return final_lung_mask, processed_image, lung_bbox, processed_left, processed_right

        except Exception as e:
            if self.debug_mode:
                print(f"      ❌ Lung segmentation failed: {e}")
                import traceback
                traceback.print_exc()
            # 返回原图
            return np.ones_like(image_8bit, dtype=bool), image_8bit, None, None, None

    def _refine_lung_mask(self, lung_mask, side, filename_debug):
        """精细化肺掩码"""
        if not np.any(lung_mask):
            return lung_mask

        try:
            # 参数设置
            image_size = max(lung_mask.shape)
            r_ball = max(10, int(image_size * 0.05))
            r_disk = max(2, int(r_ball / 6))

            if self.debug_mode:
                print(f"        {side} lung refinement: r_ball={r_ball}, r_disk={r_disk}")

            # 步骤1: 反转掩码
            inverted = ~lung_mask

            # 步骤2: 开运算
            kernel_open = morphology.ellipse(r_ball // 2, r_ball // 2)
            opened = morphology.opening(inverted, kernel_open)

            # 步骤3: 再次反转得到模糊掩码
            blurred_mask = ~opened

            # 步骤4: 二值化
            refined_mask = blurred_mask.astype(bool)

            # 步骤5: 填充空洞
            filled_mask = ndimage.binary_fill_holes(refined_mask)

            # 步骤6: 腐蚀操作平滑边界
            kernel_erode = morphology.disk(r_disk)
            final_mask = morphology.erosion(filled_mask, kernel_erode)

            # 步骤7: 凸包处理
            try:
                convex_hull = morphology.convex_hull_image(final_mask)
                diff = convex_hull & (~final_mask)

                if np.sum(diff) > 0:
                    labeled_diff = measure.label(diff)
                    diff_regions = measure.regionprops(labeled_diff)

                    for region in diff_regions:
                        area_threshold = max(50, int(np.sum(final_mask) * 0.02))
                        if region.area > area_threshold and region.eccentricity < 0.8:
                            final_mask[labeled_diff == region.label] = True

            except Exception as e:
                if self.debug_mode:
                    print(f"          Convex hull processing failed: {e}")

            self._debug_save((final_mask * 255).astype(np.uint8),
                             f"{filename_debug}_refine_{side}", f"{side.capitalize()} Lung Refined")

            return final_mask

        except Exception as e:
            if self.debug_mode:
                print(f"        {side} lung refinement failed: {e}")
            return lung_mask

    def enhance_lung_contrast(self, lung_image):
        """增强肺部对比度"""
        if lung_image.dtype != np.uint8:
            lung_image = lung_image.astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(lung_image)

        return enhanced

    def process_8bit_image(self, image_8bit, filename_debug=""):
        """处理8位图像的完整流程"""
        if self.debug_mode:
            print(f"  🔄 Starting 8-bit lung segmentation processing")

        result = {
            'processed_image': image_8bit,
            'lung_mask': None,
            'lung_bbox': None,
            'left_lung_mask': None,
            'right_lung_mask': None,
            'success': False,
            'processing_notes': []
        }

        try:
            # 步骤1: 肺分割
            lung_mask, segmented_image, lung_bbox, left_mask, right_mask = self.segment_lungs_8bit(
                image_8bit, filename_debug
            )

            result['lung_mask'] = lung_mask
            result['lung_bbox'] = lung_bbox
            result['left_lung_mask'] = left_mask
            result['right_lung_mask'] = right_mask
            result['processing_notes'].append("Lung segmentation completed")

            # 步骤2: 对比度增强
            enhanced_image = self.enhance_lung_contrast(segmented_image)
            result['processed_image'] = enhanced_image
            result['processing_notes'].append("Contrast enhancement completed")

            result['success'] = True

            if self.debug_mode:
                print(f"    ✅ 8-bit processing completed: {' → '.join(result['processing_notes'])}")

        except Exception as e:
            result['processing_notes'].append(f"Processing failed: {e}")
            if self.debug_mode:
                print(f"    ❌ 8-bit processing failed: {e}")

        return result


# 测试代码
if __name__ == "__main__":
    from pathlib import Path
    import pydicom

    # 创建肺分割处理器
    lung_processor = LungSegmentationPreprocessor()
    lung_processor.enable_debug(save_images=True, output_dir="debug_lung_segmentation", show_comparison=True)

    # 测试CT图像
    ct_file = Path(
        r"D:\python_project\NEU-AI-Integrated-Practical-Training\lung_nodule_detect\Preprocess\LIDC-IDRI-Preprocessing\LIDC-IDRI-Test\LIDC-IDRI-0005\1.3.6.1.4.1.14519.5.2.1.6279.6001.190188259083742759886805142125\000000\000037.dcm")

    if ct_file.exists():
        print("Testing CT lung segmentation...")

        # 首先使用现有的DICOM处理器获得8位图像
        from dicom_processor_v3 import DICOMProcessorV3

        dicom_processor = DICOMProcessorV3()
        processed_result = dicom_processor.process_dicom_image(ct_file, target_size=(512, 512))

        if processed_result[0] is not None:
            processed_8bit_image = processed_result[0]  # 8位处理后的图像
            print(f"DICOM processing successful: {processed_result[2]} - {processed_result[3]}")

            # 使用肺分割处理器处理8位图像
            result = lung_processor.process_8bit_image(processed_8bit_image, ct_file.stem)

            if result['success']:
                print("✅ CT lung segmentation test successful")

                # 保存结果
                cv2.imwrite("ct_original_8bit.png", processed_8bit_image)
                cv2.imwrite("ct_lung_segmented.png", result['processed_image'])

                print(f"Processing steps: {' → '.join(result['processing_notes'])}")
            else:
                print("❌ CT lung segmentation test failed")
                print(f"Failure reason: {result['processing_notes']}")
        else:
            print("❌ DICOM processing failed")
    else:
        print(f"❌ Test file not found: {ct_file}")

    print("Lung segmentation module testing completed")