"""
修正版单张DICOM检测器 - 通过import导入现有模块
Author: veryjoyran
Date: 2025-06-24 09:12:56
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
from datetime import datetime
import json

# MonAI相关导入
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ToTensord
)
from monai.bundle import ConfigParser
from monai.networks.nets import BasicUNet
from scipy import ndimage
from skimage import morphology
import zipfile

# 🔥 导入现有的DICOM处理器模块
try:
    from improved_dicom_processor import ImprovedDicomProcessor, MultiVersionDetector
    print("✅ 成功导入现有DICOM处理器模块")
except ImportError as e:
    print(f"❌ 导入DICOM处理器失败: {e}")
    print("⚠️ 请确保improved_dicom_processor.py文件在同一目录下")
    raise

# 🔧 修复中文字体显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print(f"🔧 PyTorch版本: {torch.__version__}")
print(f"👤 当前用户: veryjoyran")
print(f"📅 当前时间: 2025-06-24 09:12:56")

# ================================
# Bundle加载器模块
# ================================

class SimpleBundleLoader2D:
    """简化的2D Bundle加载器"""

    def __init__(self, bundle_path, device='cpu'):
        self.bundle_path = Path(bundle_path)
        self.device = device
        self.model = None
        self.model_info = {}

    def load_bundle_for_single_dicom(self):
        """为单张DICOM检测加载Bundle"""
        print(f"🔄 加载Bundle (单张DICOM模式): {self.bundle_path}")

        try:
            # 解压Bundle
            if self.bundle_path.suffix.lower() == '.zip':
                bundle_dir = self._extract_bundle_zip()
            else:
                bundle_dir = self.bundle_path

            # 查找配置文件
            config_file = self._find_config_file(bundle_dir)

            if config_file:
                print(f"   找到配置文件: {config_file.name}")
                success = self._load_model_with_config(config_file)

                if not success:
                    print("   使用默认2D模型...")
                    self.model = self._create_default_2d_model()
                    self._load_weights(bundle_dir)

                print("✅ 单张DICOM模式Bundle加载完成")
                return True
            else:
                print("⚠️ 未找到配置文件，使用默认2D模型")
                self.model = self._create_default_2d_model()
                self._load_weights(bundle_dir)
                return False

        except Exception as e:
            print(f"❌ Bundle加载失败: {e}")
            self.model = self._create_default_2d_model()
            return False

    def _extract_bundle_zip(self):
        """解压Bundle ZIP文件"""
        extract_dir = Path(tempfile.mkdtemp())
        with zipfile.ZipFile(self.bundle_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        bundle_dirs = [d for d in extract_dir.iterdir() if d.is_dir()]
        return bundle_dirs[0] if bundle_dirs else extract_dir

    def _find_config_file(self, bundle_dir):
        """查找配置文件"""
        config_patterns = [
            "configs/inference.json",
            "inference.json",
            "**/inference*.json"
        ]

        for pattern in config_patterns:
            config_files = list(bundle_dir.glob(pattern))
            if config_files:
                return config_files[0]
        return None

    def _load_model_with_config(self, config_file):
        """使用配置加载模型"""
        try:
            parser = ConfigParser()
            parser.read_config(str(config_file))

            if 'network_def' in parser.config:
                print("   解析3D模型并适配为2D...")
                model_3d = parser.get_parsed_content('network_def')

                # 测试是否可以直接用于2D
                self.model = self._adapt_to_2d(model_3d)

                if self.model is not None:
                    # 加载权重
                    if 'initialize' in parser.config:
                        initializer = parser.get_parsed_content('initialize')
                        if hasattr(initializer, '__call__'):
                            initializer()

                    self.model_info = {
                        'type': f"{model_3d.__class__.__name__}_2D_Adapted",
                        'original_type': model_3d.__class__.__name__,
                        'adapted_to_2d': True,
                        'pretrained': True
                    }

                    print(f"✅ 3D模型成功适配为2D: {model_3d.__class__.__name__}")
                    return True

            return False

        except Exception as e:
            print(f"⚠️ 配置模型加载失败: {e}")
            return False

    def _adapt_to_2d(self, model_3d):
        """将3D模型适配为2D"""
        try:
            # 测试2D输入
            test_input = torch.randn(1, 1, 256, 256).to(self.device)

            try:
                model_3d.eval()
                with torch.no_grad():
                    output = model_3d(test_input)
                print("   ✅ 3D模型直接支持2D输入")
                return model_3d
            except:
                pass

            # 测试单切片3D输入
            test_input_3d = torch.randn(1, 1, 1, 256, 256).to(self.device)

            try:
                model_3d.eval()
                with torch.no_grad():
                    output = model_3d(test_input_3d)
                print("   ✅ 3D模型支持单切片输入，创建包装器")
                return SingleSlice3DWrapper(model_3d)
            except:
                pass

            # 创建默认2D模型
            print("   创建默认2D模型...")
            return self._create_default_2d_model()

        except Exception as e:
            print(f"   适配失败: {e}")
            return None

    def _create_default_2d_model(self):
        """创建默认的2D模型"""
        model = BasicUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=2,
            features=(32, 64, 128, 256, 32),
            act=("LeakyReLU", {"inplace": True}),
            norm=("instance", {"affine": True}),
            dropout=0.1
        )
        print("   创建默认2D BasicUNet模型")
        return model

    def _load_weights(self, bundle_dir):
        """加载权重"""
        try:
            weight_files = list(bundle_dir.glob("**/*.pt")) + list(bundle_dir.glob("**/*.pth"))
            if weight_files:
                weight_file = weight_files[0]
                print(f"   加载权重: {weight_file.name}")

                checkpoint = torch.load(weight_file, map_location=self.device)

                if isinstance(checkpoint, dict):
                    state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
                else:
                    state_dict = checkpoint

                # 适配权重到2D
                adapted_weights = self._adapt_weights_to_2d(state_dict)

                missing_keys, unexpected_keys = self.model.load_state_dict(adapted_weights, strict=False)

                loaded_ratio = (len(self.model.state_dict()) - len(missing_keys)) / len(self.model.state_dict())

                self.model_info.update({
                    'loaded_ratio': loaded_ratio,
                    'pretrained': loaded_ratio > 0.5
                })

                print(f"   权重加载完成，成功率: {loaded_ratio:.2f}")

        except Exception as e:
            print(f"⚠️ 权重加载失败: {e}")

    def _adapt_weights_to_2d(self, state_dict_3d):
        """将3D权重适配为2D权重"""
        adapted_weights = {}

        for key, value in state_dict_3d.items():
            # 清理键名
            clean_key = key
            for prefix in ['module.', 'model.', 'network.']:
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
                    break

            if isinstance(value, torch.Tensor):
                # 处理3D卷积权重 -> 2D卷积权重
                if 'conv' in clean_key.lower() and 'weight' in clean_key and value.dim() == 5:
                    # 取中间切片
                    adapted_value = value[:, :, value.shape[2]//2, :, :]
                    adapted_weights[clean_key] = adapted_value
                else:
                    adapted_weights[clean_key] = value
            else:
                adapted_weights[clean_key] = value

        return adapted_weights

    def get_model(self):
        """获取模型"""
        if self.model is not None:
            self.model = self.model.to(self.device)
            self.model.eval()
        return self.model

    def get_model_info(self):
        """获取模型信息"""
        return self.model_info


class SingleSlice3DWrapper(torch.nn.Module):
    """单切片3D包装器"""

    def __init__(self, model_3d):
        super().__init__()
        self.model_3d = model_3d

    def forward(self, x):
        """将2D输入转换为单切片3D"""
        if x.dim() == 4:  # (B, C, H, W)
            x = x.unsqueeze(2)  # (B, C, 1, H, W)

        output = self.model_3d(x)

        # 处理输出
        if isinstance(output, torch.Tensor) and output.dim() == 5:
            output = output.squeeze(2)
        elif isinstance(output, dict):
            for key, value in output.items():
                if isinstance(value, torch.Tensor) and value.dim() == 5:
                    output[key] = value.squeeze(2)

        return output

# ================================
# 主要检测器类
# ================================

class ImprovedSingleDicomDetector:
    """改进的单张DICOM检测器 - 使用导入的DICOM处理器"""

    def __init__(self, bundle_path=None, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.bundle_loader = None
        self.model = None
        self.model_info = {}

        # 🔥 使用导入的DICOM处理器
        print("🔄 初始化导入的DICOM处理器...")
        self.dicom_processor = ImprovedDicomProcessor()

        print(f"🚀 初始化改进版单张DICOM检测器")
        print(f"   设备: {self.device}")
        print(f"   当前用户: veryjoyran")
        print(f"   时间: 2025-06-24 09:12:56")

        # 设置简化的预处理
        self.setup_minimal_transforms()

        if bundle_path:
            self.load_bundle(bundle_path)

    def setup_minimal_transforms(self):
        """设置最小化的MonAI预处理"""
        self.transforms = Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            ToTensord(keys=["image"]),
        ])

    def load_bundle(self, bundle_path):
        """加载Bundle"""
        try:
            print(f"🔄 加载Bundle: {bundle_path}")

            self.bundle_loader = SimpleBundleLoader2D(bundle_path, self.device)
            success = self.bundle_loader.load_bundle_for_single_dicom()

            self.model = self.bundle_loader.get_model()
            self.model_info = self.bundle_loader.get_model_info()

            if self.model is None:
                raise Exception("模型加载失败")

            print(f"✅ Bundle加载完成")
            print(f"   模型类型: {self.model_info.get('type', 'Unknown')}")

            return True

        except Exception as e:
            print(f"❌ Bundle加载失败: {e}")
            return False

    def detect_with_improved_preprocessing(self, dicom_path,
                                         window_center=50, window_width=350,
                                         test_all_versions=True):
        """使用改进的预处理进行检测"""
        print(f"🔍 开始改进版DICOM检测")
        print(f"   DICOM文件: {Path(dicom_path).name}")
        print(f"   窗宽窗位: C={window_center}, W={window_width}")

        if self.model is None:
            print("❌ 模型未加载")
            return None

        if test_all_versions:
            # 🔥 使用导入的MultiVersionDetector
            print("🧪 使用导入的多版本检测器...")
            multi_detector = MultiVersionDetector(self, self.dicom_processor)
            return multi_detector.test_all_preprocessing_versions(dicom_path)
        else:
            # 使用单一预处理版本
            return self._detect_single_version(dicom_path, window_center, window_width)

    def _detect_single_version(self, dicom_path, window_center, window_width):
        """使用单一预处理版本进行检测"""

        # 1. 使用导入的DICOM预处理器
        print("🔄 使用导入的DICOM预处理器...")
        processing_result = self.dicom_processor.load_and_preprocess_dicom(
            dicom_path,
            window_center=window_center,
            window_width=window_width
        )

        if processing_result is None:
            return None

        # 2. 创建临时NIfTI文件
        temp_files = self.dicom_processor.save_preprocessing_versions_as_nifti(processing_result)

        # 3. 尝试最佳版本
        best_version = 'monai_normalized'
        if best_version in temp_files:
            temp_file = temp_files[best_version]

            try:
                result = self.detect_single_dicom_from_nifti(temp_file)

                # 清理临时文件
                for temp_f in temp_files.values():
                    try:
                        Path(temp_f).unlink()
                    except:
                        pass

                if result:
                    result['processing_result'] = processing_result
                    result['preprocessing_version'] = best_version

                return result

            except Exception as e:
                print(f"❌ 检测失败: {e}")
                return None

        return None

    def detect_single_dicom_from_nifti(self, nifti_path):
        """从NIfTI文件进行检测"""
        try:
            # 使用最小化的预处理
            data_dict = {"image": nifti_path}
            data_dict = self.transforms(data_dict)

            input_tensor = data_dict["image"]

            # 确保张量格式正确
            if input_tensor.dim() == 2:
                input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
            elif input_tensor.dim() == 3:
                input_tensor = input_tensor.unsqueeze(0)
            elif input_tensor.dim() == 4 and input_tensor.shape[2] == 1:
                # 移除单一的深度维度
                input_tensor = input_tensor.squeeze(2)

            input_tensor = input_tensor.to(self.device)

            print(f"     输入张量形状: {input_tensor.shape}")
            print(f"     输入范围: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")

            # 推理
            self.model.eval()
            with torch.no_grad():
                output = self.model(input_tensor)

                # 处理输出
                if isinstance(output, dict):
                    return self._process_detection_output(output)
                elif isinstance(output, torch.Tensor):
                    return self._process_segmentation_output(output)
                else:
                    print(f"     未知输出类型: {type(output)}")
                    return None

        except Exception as e:
            print(f"     推理失败: {e}")
            return None

    def _process_detection_output(self, output):
        """处理目标检测输出"""
        try:
            boxes = output.get('boxes', [])
            scores = output.get('scores', [])
            labels = output.get('labels', [])

            print(f"     原始检测框数量: {len(boxes) if hasattr(boxes, '__len__') else 0}")

            if hasattr(boxes, '__len__') and len(boxes) > 0:
                print(f"     置信度范围: [{min(scores):.3f}, {max(scores):.3f}]")

                # 使用更低的阈值
                for threshold in [0.1, 0.05, 0.01]:
                    filtered_indices = [i for i, score in enumerate(scores) if score > threshold]

                    if filtered_indices:
                        print(f"     阈值 {threshold}: {len(filtered_indices)} 个检测")

                        return {
                            'detection_mode': True,
                            'boxes': [boxes[i] for i in filtered_indices],
                            'scores': [scores[i] for i in filtered_indices],
                            'labels': [labels[i] for i in filtered_indices] if labels else [],
                            'threshold_used': threshold,
                            'detection_count': len(filtered_indices)
                        }

                # 返回所有检测
                return {
                    'detection_mode': True,
                    'boxes': boxes,
                    'scores': scores,
                    'labels': labels,
                    'threshold_used': 0.0,
                    'detection_count': len(boxes)
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

    def _process_segmentation_output(self, output):
        """处理分割输出"""
        try:
            print(f"     分割输出形状: {output.shape}")

            # 获取概率图
            if output.shape[1] > 1:
                probs = torch.softmax(output, dim=1)
                prob_map = probs[0, 1].cpu().numpy()
            else:
                prob_map = torch.sigmoid(output[0, 0]).cpu().numpy()

            print(f"     概率图范围: [{prob_map.min():.3f}, {prob_map.max():.3f}]")

            # 使用多个阈值测试
            thresholds = [0.5, 0.3, 0.1, 0.05]

            for threshold in thresholds:
                binary_mask = prob_map > threshold

                if np.sum(binary_mask) > 0:
                    # 连通组件分析
                    labeled_array, num_features = ndimage.label(binary_mask)

                    if num_features > 0:
                        print(f"     阈值 {threshold}: {num_features} 个组件")

                        # 提取边界框
                        boxes = []
                        scores = []

                        for i in range(1, num_features + 1):
                            mask = (labeled_array == i)
                            size = np.sum(mask)

                            if size > 10:  # 最小尺寸
                                coords = np.where(mask)
                                y1, y2 = coords[0].min(), coords[0].max()
                                x1, x2 = coords[1].min(), coords[1].max()

                                region_probs = prob_map[mask]
                                avg_prob = float(region_probs.mean())

                                boxes.append([int(x1), int(y1), int(x2), int(y2)])
                                scores.append(avg_prob)

                        if boxes:
                            return {
                                'detection_mode': False,
                                'boxes': boxes,
                                'scores': scores,
                                'labels': [1] * len(boxes),
                                'threshold_used': threshold,
                                'detection_count': len(boxes),
                                'probability_map': prob_map
                            }

            # 如果所有阈值都没有结果
            return {
                'detection_mode': False,
                'boxes': [],
                'scores': [],
                'labels': [],
                'threshold_used': 0.5,
                'detection_count': 0,
                'probability_map': prob_map
            }

        except Exception as e:
            print(f"     分割输出处理失败: {e}")
            return None

    def visualize_improved_result(self, result, save_path=None):
        """可视化改进版检测结果 - 修复中文显示"""
        if not result:
            print("❌ 无结果可视化")
            return None

        # 🔧 设置matplotlib中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 如果是多版本测试结果
        if 'test_results' in result:
            return self._visualize_multi_version_result(result, save_path)

        # 单版本结果可视化
        if 'processing_result' not in result:
            print("❌ 无预处理结果可视化")
            return None

        processing_result = result['processing_result']

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Improved DICOM Detection Results', fontsize=16, fontweight='bold')

        # 显示预处理步骤 - 使用英文标题避免字体问题
        axes[0, 0].imshow(processing_result['original_array'], cmap='gray')
        axes[0, 0].set_title('Original DICOM')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(processing_result['enhanced_array'], cmap='gray')
        axes[0, 1].set_title('CLAHE Enhanced')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(processing_result['final_array'], cmap='gray')
        axes[0, 2].set_title('Final Processed')
        axes[0, 2].axis('off')

        # 显示检测结果
        detection_count = result.get('detection_count', 0)
        axes[1, 0].imshow(processing_result['final_array'], cmap='gray')
        axes[1, 0].set_title(f'Detection Results - {detection_count} Candidates')
        axes[1, 0].axis('off')

        # 绘制检测框
        boxes = result.get('boxes', [])
        scores = result.get('scores', [])

        for i, (box, score) in enumerate(zip(boxes, scores)):
            if len(box) == 4:
                x1, y1, x2, y2 = box

                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                   linewidth=2, edgecolor='red', facecolor='none')
                axes[1, 0].add_patch(rect)

                axes[1, 0].text(x1, y1-5, f'#{i+1}\n{score:.3f}',
                               color='red', fontsize=10, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))

        # 显示统计信息 - 使用英文避免字体问题
        stats_text = f"""Detection Statistics:
Candidates: {result.get('detection_count', 0)}
Mode: {'Detection' if result.get('detection_mode', False) else 'Segmentation'}
Threshold: {result.get('threshold_used', 'N/A')}
Version: {result.get('preprocessing_version', 'Default')}

Processing Parameters:
Window Width: {processing_result['preprocessing_info']['window_width']}
Window Center: {processing_result['preprocessing_info']['window_center']}"""

        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        axes[1, 1].set_title('Detection Statistics')
        axes[1, 1].axis('off')

        # 像素值分布
        axes[1, 2].hist(processing_result['original_array'].flatten(), bins=50, alpha=0.7, label='Original')
        axes[1, 2].hist(processing_result['final_array'].flatten(), bins=50, alpha=0.7, label='Processed')
        axes[1, 2].set_title('Pixel Distribution')
        axes[1, 2].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"📸 改进版检测结果保存至: {save_path}")

        return fig

    def _visualize_multi_version_result(self, result, save_path=None):
        """可视化多版本测试结果 - 修复中文显示"""
        processing_result = result['processing_result']
        test_results = result['test_results']
        best_version = result['best_version']

        # 🔧 设置matplotlib字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle('Multi-Version Preprocessing Test Results', fontsize=16, fontweight='bold')

        # 第一行：预处理步骤 - 使用英文标题
        axes[0, 0].imshow(processing_result['original_array'], cmap='gray')
        axes[0, 0].set_title('Original DICOM')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(processing_result['windowed_array'], cmap='gray')
        axes[0, 1].set_title('Window Level')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(processing_result['enhanced_array'], cmap='gray')
        axes[0, 2].set_title('CLAHE Enhanced')
        axes[0, 2].axis('off')

        axes[0, 3].imshow(processing_result['final_array'], cmap='gray')
        axes[0, 3].set_title('Final Processing')
        axes[0, 3].axis('off')

        # 第二行和第三行：版本测试结果
        version_names = list(test_results.keys())

        for i, version_name in enumerate(version_names):
            if i >= 8:  # 最多显示8个版本
                break

            row = 1 + i // 4
            col = i % 4

            test_result = test_results[version_name]

            # 显示版本名和结果 - 使用英文和符号避免字体问题
            title = f"{version_name}\n"
            if test_result['success']:
                title += f"✓ {test_result['detection_count']} detections\nConf: {test_result['max_confidence']:.3f}"
                color = 'lightgreen'
            else:
                title += "✗ No detection"
                color = 'lightcoral'

            axes[row, col].text(0.5, 0.5, title, transform=axes[row, col].transAxes,
                               ha='center', va='center', fontsize=10,
                               bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
            axes[row, col].set_title(version_name, fontsize=8)
            axes[row, col].axis('off')

        # 隐藏多余的子图
        for i in range(len(version_names), 8):
            row = 1 + i // 4
            col = i % 4
            axes[row, col].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"📸 多版本测试结果保存至: {save_path}")

        return fig

    def generate_improved_report(self, result, dicom_path):
        """生成改进版检测报告"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report = f"""
🎯 Improved Single DICOM Lung Nodule Detection Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

👤 User: veryjoyran
📅 Detection Time: {current_time}
🤖 Model: {self.model_info.get('type', 'Unknown')}
📁 DICOM File: {Path(dicom_path).name}

🔧 Improved Features (Using Imported Modules):
  • Imported DICOM processor from improved_dicom_processor
  • Reuse of YOLO successful preprocessing modules
  • Multi-version preprocessing testing via imported MultiVersionDetector
  • CLAHE contrast enhancement
"""

        # 根据结果类型生成不同的报告内容
        if 'test_results' in result:
            # 多版本测试报告
            test_results = result['test_results']
            best_version = result['best_version']

            report += f"""
📊 Multi-Version Test Results (via imported MultiVersionDetector):
  • Tested versions: {len(test_results)}
  • Best version: {best_version if best_version else 'None'}
  • Successful versions: {sum(1 for r in test_results.values() if r['success'])}

📋 Version Details:
"""

            for version_name, test_result in test_results.items():
                status = "✅" if test_result['success'] else "❌"
                count = test_result['detection_count']
                confidence = test_result['max_confidence']

                report += f"""
{status} {version_name}:
  • Detection count: {count}
  • Max confidence: {confidence:.3f}
"""
        else:
            # 单版本测试报告
            detection_count = result.get('detection_count', 0)
            report += f"""
📊 Single Version Detection Results:
  • Detected candidates: {detection_count}
  • Detection mode: {'Target Detection' if result.get('detection_mode', False) else 'Segmentation'}
  • Threshold used: {result.get('threshold_used', 'N/A')}
"""

        report += f"""

✅ Import Module Advantages:
  • Clean code architecture with modular design
  • Reuse of tested and proven DICOM processing logic
  • Easy maintenance and updates
  • Separation of concerns

⚙️ Technical Info:
  • Using imported ImprovedDicomProcessor class
  • Using imported MultiVersionDetector class
  • Font display issues resolved with matplotlib configuration
  • Current user: veryjoyran
  • Timestamp: 2025-06-24 09:12:56

📞 Technical Support: veryjoyran | Modular Version: v3.0.0
"""

        return report


# 🔧 创建简化的Gradio接口适配器
def create_gradio_compatible_detector():
    """创建与Gradio兼容的检测器实例"""
    try:
        print("🔄 创建Gradio兼容的检测器...")
        detector = ImprovedSingleDicomDetector()
        print("✅ Gradio兼容检测器创建成功")
        return detector
    except Exception as e:
        print(f"❌ Gradio兼容检测器创建失败: {e}")
        return None


# 使用示例
if __name__ == "__main__":
    print("🚀 使用导入模块的DICOM检测器测试")
    print(f"👤 当前用户: veryjoyran")
    print(f"📅 当前时间: 2025-06-24 09:12:56")
    print("=" * 60)

    # 测试模块导入
    try:
        print("🧪 测试导入的模块...")
        test_processor = ImprovedDicomProcessor()
        print("✅ ImprovedDicomProcessor导入成功")

        # 示例使用
        bundle_path = "lung_nodule_ct_detection_v0.5.4.zip"
        dicom_path = "sample.dcm"

        if Path(bundle_path).exists() and Path(dicom_path).exists():
            detector = ImprovedSingleDicomDetector(bundle_path)

            if detector.model is not None:
                result = detector.detect_with_improved_preprocessing(
                    dicom_path,
                    window_center=50,
                    window_width=350,
                    test_all_versions=True
                )

                if result:
                    # 可视化
                    fig = detector.visualize_improved_result(result, "imported_detection_result.png")

                    # 生成报告
                    report = detector.generate_improved_report(result, dicom_path)
                    print(report)

                    print("✅ 使用导入模块的检测完成!")
                else:
                    print("❌ 检测失败")
        else:
            print("❌ 请检查文件路径")
            print(f"   Bundle: {bundle_path}")
            print(f"   DICOM: {dicom_path}")

    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        print("请确保improved_dicom_processor.py文件在同一目录下")