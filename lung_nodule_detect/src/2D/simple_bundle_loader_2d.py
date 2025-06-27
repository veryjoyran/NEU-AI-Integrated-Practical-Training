"""
简化的2D Bundle加载器 - 专门处理单张DICOM
Author: veryjoyran
Date: 2025-06-24 03:25:15
"""

import torch
import json
from pathlib import Path
from monai.bundle import ConfigParser
from monai.networks.nets import BasicUNet
import zipfile
import tempfile


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
        """使用配置加载模型并适配为2D"""
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

            # 创建对应的2D模型
            print("   创建对应的2D模型...")
            return self._create_corresponding_2d_model(model_3d)

        except Exception as e:
            print(f"   适配失败: {e}")
            return None

    def _create_corresponding_2d_model(self, model_3d):
        """根据3D模型创建对应的2D模型"""
        model_type = model_3d.__class__.__name__

        if 'RetinaNet' in model_type:
            try:
                from monai.apps.detection.networks.retinanet_network import RetinaNet
                return RetinaNet(
                    spatial_dims=2,
                    n_input_channels=1,
                    num_classes=1,
                    conv1_t_size=[7, 7],
                    conv1_t_stride=[2, 2],
                    returned_layers=[2, 3, 4]
                )
            except:
                pass

        # 默认使用2D BasicUNet
        return self._create_default_2d_model()

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
                    adapted_value = value[:, :, value.shape[2] // 2, :, :]
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