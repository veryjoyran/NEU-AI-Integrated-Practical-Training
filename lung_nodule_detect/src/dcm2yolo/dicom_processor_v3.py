import pydicom
import numpy as np
import cv2


class DICOMProcessorV3:
    def get_sop_uid(self, dicom_filepath):
        try:
            dcm = pydicom.dcmread(dicom_filepath, stop_before_pixels=True)
            return getattr(dcm, 'SOPInstanceUID', None)
        except Exception as e:
            return None

    def _normalize_to_8bit(self, image_data_float, percentile_clip=True, p_low=1.0, p_high=99.0):
        """
        将浮点数组归一化到0-255 uint8

        Args:
            image_data_float: 输入的浮点数组
            percentile_clip: 是否使用百分位裁剪来处理异常值
            p_low, p_high: 百分位数的上下限
        """
        if percentile_clip:
            min_val = np.percentile(image_data_float, p_low)
            max_val = np.percentile(image_data_float, p_high)
            # 裁剪到百分位范围
            clipped_data = np.clip(image_data_float, min_val, max_val)
        else:
            clipped_data = image_data_float
            min_val = np.min(clipped_data)
            max_val = np.max(clipped_data)

        if max_val > min_val:
            normalized = ((clipped_data - min_val) / (max_val - min_val)) * 255.0
            return normalized.astype(np.uint8)
        else:
            # 如果数据平坦，返回中灰色而不是黑色，便于调试
            return np.full_like(clipped_data, 128, dtype=np.uint8)

    def _process_ct_image(self, dcm, pixel_array, filename_for_debug):
        """CT图像特定的处理逻辑"""
        try:
            # HU转换
            intercept = float(getattr(dcm, 'RescaleIntercept', 0))
            slope = float(getattr(dcm, 'RescaleSlope', 1))
            hu_array = pixel_array.astype(np.float64) * slope + intercept

            print(f"CT Processing ({filename_for_debug}): HU range {hu_array.min():.1f} to {hu_array.max():.1f}")

            # 策略1: 使用DICOM标签中的窗宽窗位
            wc_dcm = getattr(dcm, 'WindowCenter', None)
            ww_dcm = getattr(dcm, 'WindowWidth', None)

            if isinstance(wc_dcm, pydicom.multival.MultiValue):
                wc_dcm = float(wc_dcm[0]) if len(wc_dcm) > 0 else None
            elif wc_dcm is not None:
                wc_dcm = float(wc_dcm)

            if isinstance(ww_dcm, pydicom.multival.MultiValue):
                ww_dcm = float(ww_dcm[0]) if len(ww_dcm) > 0 else None
            elif ww_dcm is not None:
                ww_dcm = float(ww_dcm)

            if wc_dcm is not None and ww_dcm is not None and ww_dcm > 0:
                print(f"  Trying DICOM window: WC={wc_dcm}, WW={ww_dcm}")
                wmin = wc_dcm - ww_dcm / 2
                wmax = wc_dcm + ww_dcm / 2
                windowed = np.clip(hu_array, wmin, wmax)
                result = self._normalize_to_8bit(windowed, percentile_clip=False)
                if np.std(result) > 10:  # 检查是否有足够的对比度
                    print(f"  DICOM window successful, std={np.std(result):.1f}")
                    return result, "DICOM_Window"

            # 策略2: 标准肺窗
            print(f"  Trying lung window: WC=-600, WW=1500")
            windowed = np.clip(hu_array, -1350, 150)  # 肺窗: WC=-600, WW=1500
            result = self._normalize_to_8bit(windowed, percentile_clip=False)
            if np.std(result) > 10:
                print(f"  Lung window successful, std={np.std(result):.1f}")
                return result, "Lung_Window"

            # 策略3: 自适应HU窗口（基于HU值分布）
            print(f"  Trying adaptive HU window")
            hu_p1, hu_p99 = np.percentile(hu_array, [1, 99])
            if hu_p99 > hu_p1:
                windowed = np.clip(hu_array, hu_p1, hu_p99)
                result = self._normalize_to_8bit(windowed, percentile_clip=False)
                if np.std(result) > 10:
                    print(f"  Adaptive HU window successful, std={np.std(result):.1f}")
                    return result, "Adaptive_HU"

            # 策略4: 全HU范围归一化
            print(f"  Using full HU range normalization")
            result = self._normalize_to_8bit(hu_array, percentile_clip=True)
            return result, "Full_HU_Range"

        except Exception as e:
            print(f"CT processing error for {filename_for_debug}: {e}")
            return None, "CT_Error"

    def _process_xray_image(self, dcm, pixel_array, filename_for_debug):
        """X光片特定的处理逻辑"""
        try:
            print(f"X-Ray Processing ({filename_for_debug}): Pixel range {pixel_array.min()} to {pixel_array.max()}")

            # 检查PhotometricInterpretation
            photometric = getattr(dcm, 'PhotometricInterpretation', 'MONOCHROME2')
            print(f"  PhotometricInterpretation: {photometric}")

            pixel_data = pixel_array.astype(np.float64)

            # 策略1: 直接百分位归一化（适用于大多数X光片）
            result = self._normalize_to_8bit(pixel_data, percentile_clip=True, p_low=2, p_high=98)
            if np.std(result) > 15:  # X光片通常有更高的对比度
                print(f"  Percentile normalization successful, std={np.std(result):.1f}")
                return result, "XRay_Percentile"

            # 策略2: 如果是MONOCHROME1，尝试反转
            if photometric == 'MONOCHROME1':
                print(f"  Trying MONOCHROME1 inversion")
                inverted_data = np.max(pixel_data) - pixel_data
                result = self._normalize_to_8bit(inverted_data, percentile_clip=True, p_low=2, p_high=98)
                if np.std(result) > 15:
                    print(f"  MONOCHROME1 inversion successful, std={np.std(result):.1f}")
                    return result, "XRay_Inverted"

            # 策略3: 全范围归一化
            print(f"  Using full range normalization")
            min_val, max_val = np.min(pixel_data), np.max(pixel_data)
            if max_val > min_val:
                result = ((pixel_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                return result, "XRay_Full_Range"

            # 最后的回退
            return np.full_like(pixel_data, 128, dtype=np.uint8), "XRay_Fallback"

        except Exception as e:
            print(f"X-Ray processing error for {filename_for_debug}: {e}")
            return None, "XRay_Error"

    def process_dicom_image(self, dicom_filepath, target_size=(512, 512)):
        """
        主处理函数：自动检测模态并应用相应处理

        Returns:
            tuple: (processed_image, original_shape, modality, strategy)
        """
        try:
            dcm = pydicom.dcmread(dicom_filepath)
            filename_for_debug = dicom_filepath.name

            # 获取模态信息
            modality = getattr(dcm, 'Modality', 'UNKNOWN').upper()
            pixel_array = dcm.pixel_array
            original_shape = pixel_array.shape

            print(f"\nProcessing {filename_for_debug}: Modality={modality}, Shape={original_shape}")

            # 根据模态选择处理策略
            if modality == 'CT':
                processed_image, strategy = self._process_ct_image(dcm, pixel_array, filename_for_debug)
            elif modality in ['CR', 'DX', 'DR', 'CXR']:  # 各种X光片模态
                processed_image, strategy = self._process_xray_image(dcm, pixel_array, filename_for_debug)
            else:
                print(f"  Unknown modality {modality}, trying generic processing")
                # 对未知模态尝试通用处理
                processed_image = self._normalize_to_8bit(pixel_array.astype(np.float64), percentile_clip=True)
                strategy = f"Generic_{modality}"

            if processed_image is None:
                print(f"  All processing strategies failed, creating fallback image")
                processed_image = np.full(original_shape, 128, dtype=np.uint8)
                strategy = "Fallback_Gray"

            # 调整大小
            if target_size and processed_image.shape != target_size:
                cv2_target_size = (target_size[1], target_size[0])  # cv2需要(width, height)
                resized_image = cv2.resize(processed_image, cv2_target_size, interpolation=cv2.INTER_LINEAR)
            else:
                resized_image = processed_image

            print(
                f"  Final result: {strategy}, output range [{resized_image.min()}, {resized_image.max()}], std={np.std(resized_image):.1f}")

            return resized_image, original_shape, modality, strategy

        except Exception as e:
            print(f"CRITICAL Error processing {dicom_filepath.name}: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None, None


# 测试代码
if __name__ == '__main__':
    from pathlib import Path

    processor = DICOMProcessorV3()

    # 测试问题文件
    test_files = [
        r"D:\python_project\NEU-AI-Integrated-Practical-Training\lung_nodule_detect\Preprocess\LIDC-IDRI-Preprocessing\LIDC-IDRI\LIDC-IDRI-0001\1.3.6.1.4.1.14519.5.2.1.6279.6001.175012972118199124641098335511\000000\000000.dcm",
        r"D:\python_project\NEU-AI-Integrated-Practical-Training\lung_nodule_detect\Preprocess\LIDC-IDRI-Preprocessing\LIDC-IDRI\LIDC-IDRI-0001\1.3.6.1.4.1.14519.5.2.1.6279.6001.175012972118199124641098335511\000000\000001.dcm"
    ]

    for test_path in test_files:
        dcm_file = Path(test_path)
        if dcm_file.exists():
            print(f"\n{'=' * 60}")
            print(f"Testing: {dcm_file.name}")
            image, orig_shape, modality, strategy = processor.process_dicom_image(dcm_file, target_size=(512, 512))
            if image is not None:
                print(f"SUCCESS: {dcm_file.name} -> Modality={modality}, Strategy={strategy}")
                cv2.imwrite(f"test_output_{dcm_file.stem}_{modality}_{strategy}.png", image)
                print(f"Saved test output as test_output_{dcm_file.stem}_{modality}_{strategy}.png")
            else:
                print(f"FAILED: {dcm_file.name}")
        else:
            print(f"File not found: {test_path}")