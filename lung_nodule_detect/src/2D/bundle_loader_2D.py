"""
2D版本的MonAI Bundle加载器
Author: veryjoyran
Date: 2025-06-24 03:04:13
"""

import torch
import json
import yaml
from pathlib import Path
from monai.bundle import ConfigParser
from monai.networks.nets import BasicUNet, UNETR
import zipfile
import tempfile


class Bundle2DLoader:
    """2D专用的Bundle加载器"""

    def __init__(self, bundle_path, device='cpu'):
        self.bundle_path = Path(bundle_path)
        self.device = device
        self.model = None
        self.config = None
        self.model_info = {}
        self.bundle_dir = None

    def load_bundle_for_2d(self):
        """为2D推理加载Bundle"""
        print(f"🔄 加载MonAI Bundle (2D模式): {self.bundle_path}")

        try:
            # 处理ZIP文件
            if self.bundle_path.suffix.lower() == '.zip':
                self.bundle_dir = self._extract_bundle_zip()
            else:
                self.bundle_dir = self.bundle_path

            print(f"   Bundle目录: {self.bundle_dir}")

            # 查找配置文件
            config_file = self._find_config_file(self.bundle_dir)

            if config_file:
                print(f"   找到配置文件: {config_file.name}")
                self._load_config(config_file)

                # 🔥 使用ConfigParser加载模型
                success = self._load_model_with_config_parser_2d(config_file)

                if not success:
                    # 备用方案：手动创建2D模型
                    print("   使用备用方案创建2D模型...")
                    self.model = self._create_2d_model_from_config()
                    self._load_model_weights(self.bundle_dir)

                print("✅ 2D Bundle加载完成")
                return True
            else:
                print("⚠️ 未找到配置文件，使用默认2D模型")
                self.model = self._create_default_2d_model()
                self._try_load_weights(self.bundle_dir)
                return False

        except Exception as e:
            print(f"❌ 2D Bundle加载失败: {e}")
            import traceback
            traceback.print_exc()
            self.model = self._create_default_2d_model()
            return False

    def _extract_bundle_zip(self):
        """解压Bundle ZIP文件"""
        extract_dir = Path(tempfile.mkdtemp())

        with zipfile.ZipFile(self.bundle_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        # 查找实际的Bundle目录
        bundle_dirs = [d for d in extract_dir.iterdir() if d.is_dir()]
        if bundle_dirs:
            return bundle_dirs[0]
        else:
            return extract_dir

    def _find_config_file(self, bundle_dir):
        """查找配置文件"""
        config_patterns = [
            "configs/inference.json",
            "configs/inference.yaml",
            "configs/train.json",
            "configs/train.yaml",
            "inference.json",
            "train.json"
        ]

        for pattern in config_patterns:
            config_files = list(bundle_dir.glob(pattern))
            if config_files:
                return config_files[0]

        return None

    def _load_config(self, config_file):
        """加载配置文件"""
        try:
            if config_file.suffix.lower() == '.json':
                with open(config_file, 'r') as f:
                    self.config = json.load(f)
            else:
                with open(config_file, 'r') as f:
                    self.config = yaml.safe_load(f)

            print(f"   配置加载成功，包含 {len(self.config)} 个顶级键")

        except Exception as e:
            print(f"⚠️ 配置文件加载失败: {e}")
            self.config = {}

    def _load_model_with_config_parser_2d(self, config_file):
        """使用ConfigParser加载模型并适配2D"""
        try:
            print("🔄 使用ConfigParser解析Bundle (2D适配)...")

            # 使用MonAI的ConfigParser
            parser = ConfigParser()
            parser.read_config(str(config_file))

            # 尝试获取网络
            network_keys = ['network_def', 'network', 'model', 'detector']
            network_config = None

            for key in network_keys:
                if key in parser.config:
                    print(f"   使用配置键: {key}")
                    try:
                        network_config = parser.get_parsed_content(key)
                        break
                    except Exception as e:
                        print(f"   配置键 {key} 解析失败: {e}")
                        continue

            if network_config is not None:
                print(f"   成功解析网络配置: {type(network_config)}")

                # 🔥 关键：适配3D模型到2D
                self.model = self._adapt_3d_model_to_2d(network_config)

                if self.model is None:
                    print("   3D到2D适配失败，使用备用方案")
                    return False

                # 加载权重
                success = self._load_model_weights_with_parser(self.bundle_dir, parser)

                # 设置模型信息
                self.model_info = {
                    'type': f"{network_config.__class__.__name__}_2D_Adapted",
                    'original_type': network_config.__class__.__name__,
                    'pretrained': success,
                    'bundle_path': str(self.bundle_path),
                    'config_parser': True,
                    'adapted_to_2d': True
                }

                print(f"✅ 2D适配成功: {self.model.__class__.__name__}")
                return True
            else:
                print("⚠️ ConfigParser未找到网络配置")
                return False

        except Exception as e:
            print(f"⚠️ ConfigParser 2D适配失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _adapt_3d_model_to_2d(self, model_3d):
        """将3D模型适配为2D模型"""
        try:
            print("🔧 尝试将3D模型适配为2D...")

            # 方法1: 直接测试2D输入
            test_input_2d = torch.randn(1, 1, 256, 256).to(self.device)

            try:
                model_3d.eval()
                with torch.no_grad():
                    output = model_3d(test_input_2d)
                print("   ✅ 3D模型直接支持2D输入")
                return model_3d.to(self.device)
            except Exception as e:
                print(f"   直接2D输入失败: {e}")

            # 方法2: 使用单切片3D输入
            test_input_3d_single = torch.randn(1, 1, 1, 256, 256).to(self.device)

            try:
                model_3d.eval()
                with torch.no_grad():
                    output = model_3d(test_input_3d_single)
                print("   ✅ 3D模型支持单切片3D输入 (可用于2D)")

                # 创建包装器
                wrapper = Model3DTo2DWrapper(model_3d)
                return wrapper.to(self.device)

            except Exception as e:
                print(f"   单切片3D输入也失败: {e}")

            # 方法3: 创建对应的2D模型
            print("   尝试创建对应的2D版本...")
            model_2d = self._create_corresponding_2d_model(model_3d)

            if model_2d is not None:
                return model_2d.to(self.device)

            return None

        except Exception as e:
            print(f"❌ 3D到2D适配失败: {e}")
            return None

    def _create_corresponding_2d_model(self, model_3d):
        """根据3D模型创建对应的2D模型"""
        try:
            model_type = model_3d.__class__.__name__
            print(f"   为 {model_type} 创建2D版本...")

            if 'RetinaNet' in model_type:
                # 为RetinaNet创建2D版本
                try:
                    from monai.apps.detection.networks.retinanet_network import RetinaNet

                    model_2d = RetinaNet(
                        spatial_dims=2,  # 🔥 关键：设置为2D
                        n_input_channels=1,
                        num_classes=1,
                        conv1_t_size=[7, 7],  # 2D卷积核
                        conv1_t_stride=[2, 2],  # 2D步长
                        returned_layers=[2, 3, 4],
                        num_anchors=3,
                        aspect_ratios=[0.5, 1.0, 2.0],
                        anchor_sizes=[[32], [64], [128]]
                    )
                    print("   ✅ 创建2D RetinaNet成功")
                    return model_2d

                except Exception as e:
                    print(f"   2D RetinaNet创建失败: {e}")

            elif 'BasicUNet' in model_type:
                # 为BasicUNet创建2D版本
                model_2d = BasicUNet(
                    spatial_dims=2,  # 🔥 关键：设置为2D
                    in_channels=1,
                    out_channels=2,
                    features=(32, 64, 128, 256, 512, 32),
                    act=("LeakyReLU", {"inplace": True}),
                    norm=("instance", {"affine": True}),
                    dropout=0.1
                )
                print("   ✅ 创建2D BasicUNet成功")
                return model_2d

            elif 'UNETR' in model_type:
                # 为UNETR创建2D版本
                model_2d = UNETR(
                    in_channels=1,
                    out_channels=2,
                    img_size=(256, 256),  # 2D图像尺寸
                    feature_size=16,
                    hidden_size=768,
                    mlp_dim=3072,
                    num_heads=12,
                    spatial_dims=2  # 可能需要此参数
                )
                print("   ✅ 创建2D UNETR成功")
                return model_2d

            return None

        except Exception as e:
            print(f"   对应2D模型创建失败: {e}")
            return None

    def _create_default_2d_model(self):
        """创建默认的2D模型"""
        model = BasicUNet(
            spatial_dims=2,  # 2D
            in_channels=1,
            out_channels=2,
            features=(32, 64, 128, 256, 512, 32),
            act=("LeakyReLU", {"inplace": True}),
            norm=("instance", {"affine": True}),
            dropout=0.1
        )

        print("   创建默认2D BasicUNet模型")
        return model

    def _create_2d_model_from_config(self):
        """根据配置创建2D模型"""
        try:
            if 'network_def' in self.config:
                network_config = self.config['network_def']
            elif 'network' in self.config:
                network_config = self.config['network']
            else:
                return self._create_default_2d_model()

            if isinstance(network_config, dict):
                net_type = network_config.get('_target_', '').split('.')[-1]
                print(f"   检测到网络类型: {net_type}")

                if 'RetinaNet' in net_type:
                    return self._create_2d_retinanet(network_config)
                elif 'BasicUNet' in net_type:
                    return self._create_2d_basic_unet(network_config)
                else:
                    return self._create_default_2d_model()
            else:
                return self._create_default_2d_model()

        except Exception as e:
            print(f"⚠️ 从配置创建2D模型失败: {e}")
            return self._create_default_2d_model()

    def _create_2d_retinanet(self, config):
        """创建2D RetinaNet模型"""
        try:
            from monai.apps.detection.networks.retinanet_network import RetinaNet

            params = {
                'spatial_dims': 2,  # 强制2D
                'n_input_channels': 1,
                'num_classes': 1,
                'conv1_t_size': [7, 7],  # 2D卷积核
                'conv1_t_stride': [2, 2],  # 2D步长
                'returned_layers': [2, 3, 4],
                'num_anchors': 3,
                'aspect_ratios': [0.5, 1.0, 2.0],
                'anchor_sizes': [[32], [64], [128]]
            }

            print(f"   创建2D RetinaNet: {params}")
            return RetinaNet(**params)

        except Exception as e:
            print(f"⚠️ 2D RetinaNet创建失败: {e}")
            return self._create_default_2d_model()

    def _create_2d_basic_unet(self, config):
        """创建2D BasicUNet模型"""
        params = {
            'spatial_dims': 2,  # 强制2D
            'in_channels': 1,
            'out_channels': 2,
            'features': (32, 64, 128, 256, 512, 32),
            'act': ("LeakyReLU", {"inplace": True}),
            'norm': ("instance", {"affine": True}),
            'dropout': 0.1
        }

        print(f"   创建2D BasicUNet: {params}")
        return BasicUNet(**params)

    def _load_model_weights_with_parser(self, bundle_dir, parser):
        """使用ConfigParser加载权重"""
        try:
            # 尝试从parser中获取权重初始化信息
            if 'initialize' in parser.config:
                print("   使用Bundle初始化配置...")
                initializer = parser.get_parsed_content('initialize')
                if hasattr(initializer, '__call__'):
                    initializer()
                    print("   Bundle权重初始化完成")
                    return True

            # 备用方案：手动加载权重文件
            return self._load_model_weights(bundle_dir)

        except Exception as e:
            print(f"⚠️ Parser权重加载失败: {e}")
            return self._load_model_weights(bundle_dir)

    def _load_model_weights(self, bundle_dir):
        """加载模型权重"""
        weight_patterns = [
            "models/model.pt",
            "models/model.pth",
            "model.pt",
            "model.pth"
        ]

        weight_file = None
        for pattern in weight_patterns:
            weight_files = list(bundle_dir.glob(pattern))
            if weight_files:
                weight_file = weight_files[0]
                break

        if weight_file:
            return self._load_weights_from_file(weight_file)
        else:
            print("⚠️ 未找到权重文件")
            return False

    def _try_load_weights(self, bundle_dir):
        """尝试加载权重（备用方案）"""
        try:
            return self._load_model_weights(bundle_dir)
        except Exception as e:
            print(f"⚠️ 权重加载失败: {e}")
            return False

    def _load_weights_from_file(self, weight_file):
        """从文件加载权重"""
        try:
            print(f"   加载权重: {weight_file.name}")

            checkpoint = torch.load(weight_file, map_location=self.device)

            # 处理不同的权重格式
            if isinstance(checkpoint, dict):
                state_dict = self._extract_state_dict(checkpoint)
            else:
                state_dict = checkpoint

            # 清理键名
            cleaned_state_dict = self._clean_state_dict_keys(state_dict)

            # 🔥 关键：处理3D权重到2D的适配
            adapted_state_dict = self._adapt_weights_3d_to_2d(cleaned_state_dict)

            # 加载权重
            missing_keys, unexpected_keys = self.model.load_state_dict(adapted_state_dict, strict=False)

            print(f"   权重加载完成")

            # 计算加载比例
            total_model_keys = len(self.model.state_dict())
            loaded_keys = total_model_keys - len(missing_keys)
            loaded_ratio = loaded_keys / total_model_keys if total_model_keys > 0 else 0.0

            print(f"   模型总参数: {total_model_keys}")
            print(f"   成功加载: {loaded_keys}")
            print(f"   加载比例: {loaded_ratio:.2f}")
            print(f"   缺少键: {len(missing_keys)}")
            print(f"   多余键: {len(unexpected_keys)}")

            success = loaded_ratio > 0.5  # 50%以上认为成功

            self.model_info.update({
                'loaded_ratio': loaded_ratio,
                'missing_keys_count': len(missing_keys),
                'unexpected_keys_count': len(unexpected_keys),
                'total_params': total_model_keys,
                'pretrained': success
            })

            return success

        except Exception as e:
            print(f"❌ 权重加载失败: {e}")
            return False

    def _adapt_weights_3d_to_2d(self, state_dict_3d):
        """将3D权重适配为2D权重"""
        try:
            print("   🔧 适配3D权重到2D...")

            adapted_state_dict = {}
            adapted_count = 0

            for key, value in state_dict_3d.items():
                if isinstance(value, torch.Tensor):

                    # 处理3D卷积权重 -> 2D卷积权重
                    if 'conv' in key.lower() and 'weight' in key and value.dim() == 5:
                        # 3D卷积权重: (out_channels, in_channels, depth, height, width)
                        # 取中间切片作为2D权重: (out_channels, in_channels, height, width)
                        depth_dim = value.shape[2]
                        middle_slice = depth_dim // 2
                        adapted_value = value[:, :, middle_slice, :, :]
                        adapted_state_dict[key] = adapted_value
                        adapted_count += 1
                        print(f"     适配3D卷积 {key}: {value.shape} -> {adapted_value.shape}")

                    # 处理3D BatchNorm/GroupNorm等
                    elif any(norm_type in key.lower() for norm_type in ['norm', 'bn', 'gn']) and value.dim() >= 1:
                        # 归一化层的权重通常可以直接使用
                        adapted_state_dict[key] = value

                    # 处理全连接层
                    elif 'fc' in key.lower() or 'linear' in key.lower():
                        adapted_state_dict[key] = value

                    # 处理其他权重
                    else:
                        adapted_state_dict[key] = value
                else:
                    adapted_state_dict[key] = value

            print(f"   权重适配完成，适配了 {adapted_count} 个3D卷积层")
            return adapted_state_dict

        except Exception as e:
            print(f"   权重适配失败: {e}")
            return state_dict_3d  # 返回原始权重

    def _extract_state_dict(self, checkpoint):
        """从checkpoint提取state_dict"""
        possible_keys = ['model', 'state_dict', 'model_state_dict', 'network', 'detector']

        for key in possible_keys:
            if key in checkpoint:
                print(f"   使用权重键: {key}")
                return checkpoint[key]

        print("   使用完整checkpoint作为state_dict")
        return checkpoint

    def _clean_state_dict_keys(self, state_dict):
        """清理state_dict的键名"""
        cleaned_state_dict = {}

        for key, value in state_dict.items():
            cleaned_key = key
            prefixes_to_remove = ['module.', 'model.', 'network.', 'detector.']

            for prefix in prefixes_to_remove:
                if cleaned_key.startswith(prefix):
                    cleaned_key = cleaned_key[len(prefix):]
                    break

            cleaned_state_dict[cleaned_key] = value

        print(f"   清理后权重键数量: {len(cleaned_state_dict)}")
        return cleaned_state_dict

    def get_model(self):
        """获取加载的模型"""
        if self.model is not None:
            self.model = self.model.to(self.device)
            self.model.eval()
        return self.model

    def get_model_info(self):
        """获取模型信息"""
        return self.model_info


class Model3DTo2DWrapper(torch.nn.Module):
    """3D模型到2D的包装器"""

    def __init__(self, model_3d):
        super().__init__()
        self.model_3d = model_3d

    def forward(self, x):
        """将2D输入转换为单切片3D输入"""
        if x.dim() == 4:  # (B, C, H, W)
            x = x.unsqueeze(2)  # (B, C, 1, H, W)

        # 调用3D模型
        output = self.model_3d(x)

        # 如果输出是3D，压缩深度维度
        if isinstance(output, torch.Tensor) and output.dim() == 5:
            output = output.squeeze(2)  # (B, C, H, W)
        elif isinstance(output, dict):
            # 处理检测输出
            for key, value in output.items():
                if isinstance(value, torch.Tensor) and value.dim() == 5:
                    output[key] = value.squeeze(2)

        return output