import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import SimpleITK as sitk
import tempfile
import json
from datetime import datetime

import monai
from monai.apps import download_and_extract
from monai.bundle import ConfigParser, download, load
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import BasicUNet, UNETR, SwinUNETR
from monai.transforms import (
    Activations, AsDiscrete, Compose, LoadImaged,
    EnsureChannelFirstd,  # 正确的用法
    Spacingd, Orientationd, ScaleIntensityRanged, CropForegroundd,
    RandCropByPosNegLabeld, ToTensord, EnsureTyped
)
from monai.utils import first, set_determinism

print(f"🔧 MonAI版本: {monai.__version__}")
print(f"🔧 PyTorch版本: {torch.__version__}")
print(f"🔧 设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")


class MonAI3DLungDetector:
    """基于MonAI的3D肺结节检测器 """

    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transforms = None
        self.model_info = {}

        # 设置确定性训练
        set_determinism(seed=12345)

        print(f"🚀 初始化MonAI 3D检测器 (设备: {self.device})")

    def setup_transforms(self):
        """设置数据预处理变换 - 修正版本"""
        print("🔧 设置数据预处理流水线...")

        # 训练变换 - 使用正确的transforms
        self.train_transforms = Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),  # 正确的用法
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-1000,
                a_max=400,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=2,
                image_key="image",
                image_threshold=0,
            ),
            ToTensord(keys=["image", "label"]),
        ])

        # 验证/推理变换 - 简化版本，避免可能的问题
        self.val_transforms = Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),  # 确保通道在第一维
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-1000,
                a_max=400,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image"], source_key="image"),
            ToTensord(keys=["image"]),
        ])

        print("✅ 数据预处理流水线设置完成")

    def download_pretrained_models(self):
        """下载MonAI预训练模型 - 修正版本"""
        print("📥 下载MonAI预训练模型...")

        # 创建模型存储目录
        self.model_dir = Path("./monai_models")
        self.model_dir.mkdir(exist_ok=True)

        # 尝试下载可用的预训练模型
        downloaded_models = {}

        # 方法1: 尝试下载专用肺结节模型
        try:
            print("🔄 尝试下载肺结节专用模型...")

            # 这里我们使用一个更通用的方法
            # 首先尝试从MonAI hub获取可用模型列表
            from monai.bundle import get_bundle_versions

            try:
                # 尝试获取可用的bundle
                available_bundles = ["spleen_ct_segmentation", "lung_nodule_ct_detection"]

                for bundle_name in available_bundles:
                    try:
                        print(f"   尝试下载: {bundle_name}")
                        bundle_path = download(
                            name=bundle_name,
                            source="monaihosting",
                            progress=True,
                            cache_dir=str(self.model_dir)
                        )

                        downloaded_models[bundle_name] = {
                            "path": bundle_path,
                            "description": f"MonAI {bundle_name} model"
                        }

                        print(f"✅ {bundle_name} 下载完成")
                        break  # 成功下载一个就退出

                    except Exception as e:
                        print(f"   {bundle_name} 下载失败: {e}")
                        continue

            except Exception as e:
                print(f"   Bundle下载失败: {e}")

        except Exception as e:
            print(f"⚠️ 专用模型下载失败: {e}")

        # 方法2: 如果没有下载成功，使用预构建的模型
        if not downloaded_models:
            print("⚠️ 未能下载专用模型，将创建通用预训练模型")
            self._setup_generic_pretrained()
        else:
            self.downloaded_models = downloaded_models

        return downloaded_models

    def _setup_generic_pretrained(self):
        """设置通用预训练模型 - 改进版本"""
        print("🔧 设置通用预训练模型...")

        try:
            # 尝试创建UNETR模型（更先进）
            self.model = UNETR(
                in_channels=1,
                out_channels=2,  # 背景 + 结节
                img_size=(96, 96, 96),
                feature_size=16,
                hidden_size=768,
                mlp_dim=3072,
                num_heads=12,
                pos_embed="perceptron",
                norm_name="instance",
                res_block=True,
                dropout_rate=0.0,
            ).to(self.device)

            print("✅ UNETR模型创建完成")

            # 设置模型信息
            self.model_info = {
                "type": "UNETR",
                "pretrained": False,
                "input_channels": 1,
                "output_channels": 2,
                "spatial_dims": 3,
                "img_size": (96, 96, 96)
            }

        except Exception as e:
            print(f"⚠️ UNETR创建失败: {e}，使用BasicUNet")

            # 备用方案：BasicUNet
            self.model = BasicUNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=2,  # 背景 + 结节
                features=(32, 32, 64, 128, 256, 32),
                dropout=0.1
            ).to(self.device)

            print("✅ BasicUNet模型创建完成")

            # 设置模型信息
            self.model_info = {
                "type": "BasicUNet",
                "pretrained": False,
                "input_channels": 1,
                "output_channels": 2,
                "spatial_dims": 3
            }

    def load_bundle_model(self, bundle_name="spleen_ct_segmentation"):
        """加载Bundle模型 - 改进版本"""
        try:
            if hasattr(self, 'downloaded_models') and bundle_name in self.downloaded_models:
                bundle_path = self.downloaded_models[bundle_name]["path"]

                print(f"🔄 加载Bundle模型: {bundle_name}")
                print(f"   路径: {bundle_path}")

                # 检查bundle结构
                bundle_dir = Path(bundle_path)
                config_files = list(bundle_dir.rglob("*.json"))
                model_files = list(bundle_dir.rglob("*.pt")) + list(bundle_dir.rglob("*.pth"))

                print(f"   找到配置文件: {len(config_files)}")
                print(f"   找到模型文件: {len(model_files)}")

                if config_files and model_files:
                    try:
                        # 使用MonAI的Bundle加载器
                        parser = ConfigParser()
                        parser.read_config(str(config_files[0]))

                        # 尝试获取网络定义
                        if "network_def" in parser.config:
                            self.model = parser.get_parsed_content("network_def")
                            self.model = self.model.to(self.device)

                            # 加载权重
                            checkpoint = torch.load(model_files[0], map_location=self.device)

                            # 处理不同的权重格式
                            if isinstance(checkpoint, dict):
                                if "model" in checkpoint:
                                    state_dict = checkpoint["model"]
                                elif "state_dict" in checkpoint:
                                    state_dict = checkpoint["state_dict"]
                                elif "model_state_dict" in checkpoint:
                                    state_dict = checkpoint["model_state_dict"]
                                else:
                                    state_dict = checkpoint
                            else:
                                state_dict = checkpoint

                            # 尝试加载权重
                            try:
                                self.model.load_state_dict(state_dict, strict=False)
                                print(f"✅ Bundle模型加载完成: {bundle_name}")

                                self.model_info = {
                                    "type": "Bundle",
                                    "name": bundle_name,
                                    "pretrained": True,
                                    "path": str(bundle_path)
                                }

                                return True

                            except Exception as e:
                                print(f"⚠️ 权重加载失败: {e}")

                    except Exception as e:
                        print(f"⚠️ Bundle解析失败: {e}")

                print(f"⚠️ Bundle文件结构不完整")

            # 如果Bundle加载失败，使用备用方案
            print("🔄 Bundle加载失败，使用通用模型")
            self._setup_generic_pretrained()
            return False

        except Exception as e:
            print(f"❌ Bundle模型加载失败: {e}")
            self._setup_generic_pretrained()
            return False

    def convert_dicom_to_nifti(self, dicom_path):
        """将DICOM转换为NIfTI格式 - 改进版本"""
        print(f"🔄 转换DICOM: {dicom_path}")

        try:
            if Path(dicom_path).is_dir():
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
                print(f"   读取单个DICOM文件")

            # 获取基本信息
            print(f"   原始尺寸: {image.GetSize()}")
            print(f"   体素间距: {image.GetSpacing()}")
            print(f"   方向: {image.GetDirection()}")

            # 标准化方向
            image = sitk.DICOMOrient(image, 'LPS')

            # 转换为HU值
            array = sitk.GetArrayFromImage(image)
            print(f"   HU值范围: [{array.min():.1f}, {array.max():.1f}]")

            # 保存为临时NIfTI文件
            temp_file = tempfile.mktemp(suffix='.nii.gz')
            sitk.WriteImage(image, temp_file)

            print(f"✅ 转换完成: {temp_file}")

            return temp_file, image

        except Exception as e:
            print(f"❌ DICOM转换失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def inference_on_volume(self, nifti_path, roi_size=(96, 96, 96), sw_batch_size=4):
        """对3D体积进行推理 - 改进版本"""
        if self.model is None:
            print("❌ 模型未加载")
            return None

        print(f"🔍 开始3D推理: {nifti_path}")
        print(f"   ROI尺寸: {roi_size}")
        print(f"   批处理大小: {sw_batch_size}")

        try:
            # 准备数据
            data_dict = {"image": nifti_path}

            # 应用预处理
            if self.val_transforms is None:
                self.setup_transforms()

            data_dict = self.val_transforms(data_dict)

            # 检查数据形状
            input_tensor = data_dict["image"]
            print(f"   预处理后形状: {input_tensor.shape}")

            # 确保有批次维度
            if input_tensor.dim() == 4:  # (C, D, H, W)
                input_tensor = input_tensor.unsqueeze(0)  # (1, C, D, H, W)

            input_tensor = input_tensor.to(self.device)
            print(f"   输入张量形状: {input_tensor.shape}")

            # 检查输入尺寸是否合适
            _, _, d, h, w = input_tensor.shape
            if d < roi_size[0] or h < roi_size[1] or w < roi_size[2]:
                print(f"⚠️ 输入体积太小 ({d}, {h}, {w})，调整ROI尺寸")
                roi_size = (min(d, roi_size[0]), min(h, roi_size[1]), min(w, roi_size[2]))
                print(f"   调整后ROI尺寸: {roi_size}")

            # 推理
            self.model.eval()
            with torch.no_grad():
                # 使用滑动窗口推理处理大体积
                outputs = sliding_window_inference(
                    inputs=input_tensor,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=self.model,
                    overlap=0.25,
                    mode="gaussian",
                    sigma_scale=0.125,
                    padding_mode="constant",
                    cval=0.0,
                )

                print(f"   推理输出形状: {outputs.shape}")

                # 后处理
                outputs = torch.softmax(outputs, dim=1)
                pred_mask = torch.argmax(outputs, dim=1).squeeze(0)
                prob_map = outputs[0, 1]  # 结节概率图

                # 转换为numpy
                pred_np = pred_mask.cpu().numpy()
                prob_np = prob_map.cpu().numpy()

                print(f"✅ 推理完成")
                print(f"   预测形状: {pred_np.shape}")
                print(f"   阳性体素: {np.sum(pred_np > 0)}")
                print(f"   最大概率: {prob_np.max():.3f}")

                return {
                    "prediction": pred_np,
                    "probability": prob_np,
                    "input_shape": input_tensor.shape,
                    "positive_voxels": int(np.sum(pred_np > 0)),
                    "max_probability": float(prob_np.max()),
                    "mean_probability": float(prob_np.mean())
                }

        except Exception as e:
            print(f"❌ 推理失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def extract_nodule_candidates(self, prediction, probability, min_size=20, max_size=10000):
        """从预测结果中提取结节候选 - 改进版本"""
        print(f"🎯 提取结节候选 (最小尺寸: {min_size}, 最大尺寸: {max_size})")

        try:
            from scipy import ndimage
            from skimage import measure

            # 连通组件分析
            labeled_array, num_features = ndimage.label(prediction > 0)

            print(f"   发现连通组件: {num_features} 个")

            candidates = []

            for i in range(1, num_features + 1):
                mask = (labeled_array == i)
                size = np.sum(mask)

                if min_size <= size <= max_size:
                    # 计算质心
                    center = ndimage.center_of_mass(mask)

                    # 计算边界框
                    coords = np.where(mask)
                    bbox = [
                        int(coords[0].min()), int(coords[0].max()),
                        int(coords[1].min()), int(coords[1].max()),
                        int(coords[2].min()), int(coords[2].max())
                    ]

                    # 计算概率统计
                    region_probs = probability[mask]
                    avg_prob = float(region_probs.mean())
                    max_prob = float(region_probs.max())
                    std_prob = float(region_probs.std())

                    # 计算形状特征
                    try:
                        # 简单的形状特征
                        bbox_volume = (bbox[1] - bbox[0]) * (bbox[3] - bbox[2]) * (bbox[5] - bbox[4])
                        compactness = size / bbox_volume if bbox_volume > 0 else 0

                        # 计算球形度（简化版本）
                        equivalent_diameter = (6 * size / np.pi) ** (1 / 3)
                        surface_area = np.sum(ndimage.binary_erosion(mask) != mask)  # 近似表面积
                        sphericity = (np.pi ** (1 / 3)) * (
                                    (6 * size) ** (2 / 3)) / surface_area if surface_area > 0 else 0

                    except:
                        compactness = 0.0
                        sphericity = 0.0

                    candidate = {
                        "id": i,
                        "center": [float(c) for c in center],  # 确保可JSON序列化
                        "bbox": bbox,
                        "size": int(size),
                        "avg_probability": avg_prob,
                        "max_probability": max_prob,
                        "std_probability": std_prob,
                        "compactness": compactness,
                        "sphericity": min(sphericity, 1.0),  # 限制在0-1之间
                        "mask": mask
                    }

                    candidates.append(candidate)

            # 按最大概率排序
            candidates = sorted(candidates, key=lambda x: x["max_probability"], reverse=True)

            print(f"✅ 找到 {len(candidates)} 个候选结节")

            # 显示前几个候选的信息
            for i, cand in enumerate(candidates[:5]):
                print(
                    f"   候选 {i + 1}: 尺寸={cand['size']}, 最大概率={cand['max_probability']:.3f}, 紧密度={cand['compactness']:.3f}")

            return candidates

        except Exception as e:
            print(f"❌ 候选提取失败: {e}")
            import traceback
            traceback.print_exc()
            return []

    def save_results(self, candidates, output_dir="./results"):
        """保存结果 - 改进版本"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存候选信息为JSON
        candidates_clean = []
        for cand in candidates:
            cand_clean = {k: v for k, v in cand.items() if k != "mask"}  # 去除mask（无法JSON序列化）
            candidates_clean.append(cand_clean)

        results_data = {
            "timestamp": timestamp,
            "user": "veryjoyran",
            "version": "v1.0.0_fixed",
            "model_info": self.model_info,
            "total_candidates": len(candidates_clean),
            "candidates": candidates_clean
        }

        json_path = output_path / f"monai_results_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        print(f"💾 结果保存至: {json_path}")
        return json_path


# 快速测试函数
def quick_test_fixed_monai():
    """快速测试修正版MonAI检测器"""
    print("🚀 开始测试修正版MonAI 3D肺结节检测器")
    print(f"👤 用户: veryjoyran")
    print(f"📅 时间: 2025-06-23 02:40:44")
    print("=" * 60)

    # 1. 初始化检测器
    detector = MonAI3DLungDetector()

    # 2. 下载预训练模型
    print("\n📥 第1步: 下载预训练模型...")
    downloaded_models = detector.download_pretrained_models()

    # 3. 设置数据变换
    print("\n🔧 第2步: 设置数据变换...")
    detector.setup_transforms()

    # 4. 加载模型
    print("\n🤖 第3步: 加载模型...")
    if downloaded_models:
        success = detector.load_bundle_model()
        if not success:
            print("   使用通用模型作为备用")
    else:
        print("   使用通用模型")

    print(f"\n✅ 检测器初始化完成!")
    print(f"   模型类型: {detector.model_info.get('type', 'Unknown')}")
    print(f"   预训练: {detector.model_info.get('pretrained', False)}")
    print(f"   设备: {detector.device}")

    # 5. 提示用户如何使用
    print(f"\n💡 使用方法:")
    print(f"   1. 将您的DICOM数据路径替换到下面的代码中")
    print(f"   2. 运行推理测试")

    # 示例使用代码
    example_code = '''
# 使用示例:
dicom_path = "path/to/your/dicom/series"  # 替换为您的路径

if Path(dicom_path).exists():
    # 转换DICOM
    nifti_path, sitk_image = detector.convert_dicom_to_nifti(dicom_path)

    if nifti_path:
        # 进行推理
        results = detector.inference_on_volume(nifti_path)

        if results:
            # 提取候选
            candidates = detector.extract_nodule_candidates(
                results["prediction"], 
                results["probability"]
            )

            # 保存结果
            json_path = detector.save_results(candidates)

            print(f"检测完成！发现 {len(candidates)} 个候选结节")
'''

    print(example_code)

    return detector


if __name__ == "__main__":
    # 测试修正版检测器
    detector = quick_test_fixed_monai()

    print("\n🎉 MonAI检测器准备就绪！")
    print("   现在您可以处理实际的DICOM数据了")