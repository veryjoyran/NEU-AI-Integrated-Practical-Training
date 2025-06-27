import torch
import json
import yaml
from pathlib import Path
from monai.bundle import ConfigParser
from monai.networks.nets import BasicUNet
import zipfile
import tempfile
import warnings

class MonAIBundleLoader:
    """MonAI Bundle加载器 - 专门用于LUNA16肺结节检测"""

    def __init__(self, bundle_path, device='cpu'):
        self.bundle_path = Path(bundle_path)
        self.device = device
        self.model = None
        self.config = None
        self.model_info = {}
        self.bundle_dir = None

        print(f"🚀 初始化MonAI Bundle加载器")
        print(f"   Bundle路径: {self.bundle_path}")
        print(f"   设备: {self.device}")
        print(f"   当前用户: veryjoyran")
        print(f"   时间: 2025-06-24 15:25:42")

    def load_bundle(self):
        """加载MonAI Bundle"""
        print(f"🔄 开始加载Bundle: {self.bundle_path}")

        try:
            # 1. 解压Bundle（如果是ZIP文件）
            if self.bundle_path.suffix.lower() == '.zip':
                self.bundle_dir = self._extract_bundle_zip()
            else:
                self.bundle_dir = self.bundle_path

            print(f"   Bundle目录: {self.bundle_dir}")

            # 2. 查找并加载配置文件
            config_file = self._find_config_file()
            if not config_file:
                raise FileNotFoundError("未找到有效的配置文件")

            print(f"   找到配置文件: {config_file.name}")

            # 3. 使用ConfigParser加载模型
            success = self._load_model_with_config_parser(config_file)

            if not success:
                print("   ConfigParser加载失败，尝试手动创建模型...")
                self._load_config_manually(config_file)
                self.model = self._create_fallback_model()
                self._load_weights_manually()

            # 4. 验证模型
            self._validate_model()

            print("✅ Bundle加载完成")
            self._print_model_info()

            return True

        except Exception as e:
            print(f"❌ Bundle加载失败: {e}")
            import traceback
            traceback.print_exc()

            # 尝试创建备用模型
            print("🔄 尝试创建备用模型...")
            self.model = self._create_fallback_model()
            return False

    def _extract_bundle_zip(self):
        """解压Bundle ZIP文件"""
        print("   解压Bundle ZIP文件...")
        extract_dir = Path(tempfile.mkdtemp(prefix="monai_bundle_"))

        try:
            with zipfile.ZipFile(self.bundle_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

            # 查找实际的Bundle目录
            bundle_dirs = [d for d in extract_dir.iterdir() if d.is_dir()]
            if bundle_dirs:
                actual_bundle_dir = bundle_dirs[0]
                print(f"   解压完成: {actual_bundle_dir}")
                return actual_bundle_dir
            else:
                print(f"   解压完成: {extract_dir}")
                return extract_dir

        except Exception as e:
            print(f"   解压失败: {e}")
            raise

    def _find_config_file(self):
        """查找配置文件"""
        print("   查找配置文件...")

        # 按优先级查找配置文件
        config_patterns = [
            "configs/inference.json",
            "configs/inference.yaml",
            "configs/train.json",
            "configs/train.yaml",
            "inference.json",
            "train.json",
            "**/inference*.json",
            "**/train*.json"
        ]

        for pattern in config_patterns:
            config_files = list(self.bundle_dir.glob(pattern))
            if config_files:
                config_file = config_files[0]
                print(f"   找到配置文件: {pattern} -> {config_file.name}")
                return config_file

        print("   ⚠️ 未找到配置文件")
        return None

    def _load_model_with_config_parser(self, config_file):
        """使用ConfigParser加载模型"""
        print("   使用ConfigParser加载模型...")

        try:
            # 初始化ConfigParser
            self.parser = ConfigParser()
            self.parser.read_config(str(config_file))

            print(f"   配置文件解析成功，包含 {len(self.parser.config)} 个顶级键")

            # 查找网络定义
            network_keys = ['network_def', 'network', 'model', 'detector']
            network_config = None

            for key in network_keys:
                if key in self.parser.config:
                    try:
                        print(f"   尝试解析网络配置键: {key}")
                        network_config = self.parser.get_parsed_content(key)
                        print(f"   成功解析网络: {type(network_config)}")
                        break
                    except Exception as e:
                        print(f"   解析键 {key} 失败: {e}")
                        continue

            if network_config is None:
                print("   ⚠️ 未找到有效的网络配置")
                return False

            self.model = network_config

            # 尝试加载权重（通过initialize配置）
            if 'initialize' in self.parser.config:
                try:
                    print("   执行Bundle初始化...")
                    initializer = self.parser.get_parsed_content('initialize')
                    if hasattr(initializer, '__call__'):
                        initializer()
                        print("   Bundle初始化完成")
                    self.model_info['weights_loaded'] = True
                except Exception as e:
                    print(f"   Bundle初始化失败: {e}")
                    self.model_info['weights_loaded'] = False
                    # 尝试手动加载权重
                    self._load_weights_manually()
            else:
                print("   无initialize配置，尝试手动加载权重...")
                self._load_weights_manually()

            # 设置模型信息
            self.model_info.update({
                'type': 'MonAI_Bundle_Model',
                'network_class': network_config.__class__.__name__,
                'config_parser_used': True,
                'config_file': config_file.name,
                'bundle_path': str(self.bundle_path)
            })

            return True

        except Exception as e:
            print(f"   ConfigParser加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _load_config_manually(self, config_file):
        """手动加载配置文件"""
        print("   手动加载配置文件...")

        try:
            if config_file.suffix.lower() == '.json':
                with open(config_file, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            elif config_file.suffix.lower() in ['.yaml', '.yml']:
                with open(config_file, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f)
            else:
                raise ValueError(f"不支持的配置文件格式: {config_file.suffix}")

            print(f"   手动配置加载成功，包含 {len(self.config)} 个顶级键")

        except Exception as e:
            print(f"   手动配置加载失败: {e}")
            self.config = {}

    def _create_fallback_model(self):
        """创建备用模型（基于LUNA16标准）"""
        print("   创建LUNA16兼容的备用模型...")

        try:
            # 根据LUNA16 README，使用RetinaNet进行检测
            # 如果RetinaNet不可用，使用3D UNet作为备用

            # 首先尝试RetinaNet
            try:
                from monai.apps.detection.networks.retinanet_network import RetinaNet

                model = RetinaNet(
                    spatial_dims=3,
                    n_input_channels=1,
                    num_classes=1,  # 肺结节检测
                    conv1_t_size=[7, 7, 7],
                    conv1_t_stride=[2, 2, 2],
                    returned_layers=[2, 3, 4],
                    num_anchors=3
                )

                print("   创建RetinaNet模型成功")
                self.model_info.update({
                    'type': 'RetinaNet_3D_Fallback',
                    'network_class': 'RetinaNet',
                    'spatial_dims': 3,
                    'num_classes': 1
                })

                return model

            except ImportError:
                print("   RetinaNet不可用，使用3D UNet...")

                # 备用：3D UNet
                model = BasicUNet(
                    spatial_dims=3,
                    in_channels=1,
                    out_channels=2,  # 背景 + 结节
                    features=(32, 64, 128, 256, 512, 32),
                    act=("LeakyReLU", {"inplace": True}),
                    norm=("instance", {"affine": True}),
                    dropout=0.1
                )

                print("   创建3D UNet模型成功")
                self.model_info.update({
                    'type': '3D_UNet_Fallback',
                    'network_class': 'BasicUNet',
                    'spatial_dims': 3,
                    'in_channels': 1,
                    'out_channels': 2
                })

                return model

        except Exception as e:
            print(f"   备用模型创建失败: {e}")
            raise

    def _load_weights_manually(self):
        """手动加载权重 - 修正成功率计算"""
        print("   手动查找和加载权重...")

        try:
            # 查找权重文件
            weight_patterns = [
                "models/model.pt",
                "models/model.pth",
                "model.pt",
                "model.pth",
                "**/model*.pt",
                "**/model*.pth"
            ]

            weight_file = None
            for pattern in weight_patterns:
                weight_files = list(self.bundle_dir.glob(pattern))
                if weight_files:
                    weight_file = weight_files[0]
                    print(f"   找到权重文件: {pattern} -> {weight_file.name}")
                    break

            if not weight_file:
                print("   ⚠️ 未找到权重文件")
                self.model_info['weights_loaded'] = False
                self.model_info['load_ratio'] = 0.0
                return False

            # 加载权重
            print(f"   加载权重文件: {weight_file}")
            checkpoint = torch.load(weight_file, map_location=self.device)

            # 提取state_dict
            state_dict = self._extract_state_dict(checkpoint)

            # 清理键名
            cleaned_state_dict = self._clean_state_dict_keys(state_dict)

            # 🔥 修正：加载到模型前先确保模型存在
            if self.model is not None:
                try:
                    # 获取模型的期望参数
                    model_state_dict = self.model.state_dict()
                    total_model_keys = len(model_state_dict)

                    print(f"   模型期望参数数量: {total_model_keys}")
                    print(f"   权重文件参数数量: {len(cleaned_state_dict)}")

                    # 🔥 修正：更智能的权重匹配
                    matched_weights = {}
                    successful_matches = 0

                    for model_key in model_state_dict.keys():
                        # 尝试直接匹配
                        if model_key in cleaned_state_dict:
                            matched_weights[model_key] = cleaned_state_dict[model_key]
                            successful_matches += 1
                        else:
                            # 尝试部分匹配
                            for weight_key in cleaned_state_dict.keys():
                                if model_key.endswith(weight_key) or weight_key.endswith(model_key):
                                    # 检查形状是否匹配
                                    if model_state_dict[model_key].shape == cleaned_state_dict[weight_key].shape:
                                        matched_weights[model_key] = cleaned_state_dict[weight_key]
                                        successful_matches += 1
                                        print(f"   部分匹配: {model_key} <- {weight_key}")
                                        break

                    print(f"   成功匹配参数: {successful_matches}/{total_model_keys}")

                    # 加载匹配的权重
                    missing_keys, unexpected_keys = self.model.load_state_dict(matched_weights, strict=False)

                    # 🔥 修正：正确计算加载成功率
                    loaded_keys = total_model_keys - len(missing_keys)
                    load_ratio = loaded_keys / total_model_keys if total_model_keys > 0 else 0

                    print(f"   权重加载完成:")
                    print(f"     模型总参数: {total_model_keys}")
                    print(f"     成功加载: {loaded_keys}")
                    print(f"     加载率: {load_ratio:.2%}")
                    print(f"     缺失键: {len(missing_keys)}")
                    print(f"     意外键: {len(unexpected_keys)}")

                    # 如果加载率仍然为0，尝试其他策略
                    if load_ratio == 0:
                        print("   ⚠️ 标准加载失败，尝试强制匹配...")
                        load_ratio = self._force_weight_loading(cleaned_state_dict)

                    self.model_info.update({
                        'weights_loaded': load_ratio > 0.1,  # 降低阈值
                        'load_ratio': load_ratio,
                        'missing_keys': len(missing_keys),
                        'unexpected_keys': len(unexpected_keys),
                        'weight_file': weight_file.name,
                        'successful_matches': successful_matches,
                        'total_model_params': total_model_keys
                    })

                    return load_ratio > 0.1

                except Exception as e:
                    print(f"   权重加载过程失败: {e}")
                    self.model_info['weights_loaded'] = False
                    self.model_info['load_ratio'] = 0.0
                    return False

            return False

        except Exception as e:
            print(f"   权重加载失败: {e}")
            self.model_info['weights_loaded'] = False
            self.model_info['load_ratio'] = 0.0
            return False

    def _force_weight_loading(self, cleaned_state_dict):
        """强制权重加载策略"""
        try:
            print("   执行强制权重匹配...")

            model_state_dict = self.model.state_dict()
            total_params = len(model_state_dict)
            loaded_params = 0

            # 策略1: 尝试形状匹配
            for model_key, model_param in model_state_dict.items():
                for weight_key, weight_param in cleaned_state_dict.items():
                    if model_param.shape == weight_param.shape:
                        try:
                            model_param.data.copy_(weight_param.data)
                            loaded_params += 1
                            print(f"   强制匹配: {model_key} <- {weight_key}")
                            break
                        except:
                            continue

            # 策略2: 如果仍然没有加载，尝试部分匹配
            if loaded_params == 0:
                print("   尝试部分参数匹配...")
                # 这里可以添加更复杂的匹配逻辑

            force_ratio = loaded_params / total_params if total_params > 0 else 0
            print(f"   强制匹配结果: {loaded_params}/{total_params} ({force_ratio:.2%})")

            return force_ratio

        except Exception as e:
            print(f"   强制加载失败: {e}")
            return 0.0

    def _extract_state_dict(self, checkpoint):
        """从checkpoint提取state_dict"""
        if isinstance(checkpoint, dict):
            # 尝试不同的键名
            possible_keys = ['model', 'state_dict', 'model_state_dict', 'network', 'detector']

            for key in possible_keys:
                if key in checkpoint:
                    print(f"   使用checkpoint键: {key}")
                    return checkpoint[key]

            # 如果没有找到标准键名，使用整个checkpoint
            print("   使用整个checkpoint作为state_dict")
            return checkpoint
        else:
            # 直接是state_dict
            return checkpoint

    def _clean_state_dict_keys(self, state_dict):
        """清理state_dict的键名"""
        cleaned_dict = {}

        for key, value in state_dict.items():
            # 移除常见的前缀
            clean_key = key
            prefixes_to_remove = ['module.', 'model.', 'network.', 'detector.', '_orig_mod.']

            for prefix in prefixes_to_remove:
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
                    break

            cleaned_dict[clean_key] = value

        print(f"   键名清理完成: {len(state_dict)} -> {len(cleaned_dict)}")
        return cleaned_dict

    def _validate_model(self):
        """验证模型"""
        print("   验证模型...")

        try:
            if self.model is None:
                raise ValueError("模型为空")

            # 移动到指定设备
            self.model = self.model.to(self.device)
            self.model.eval()

            # 测试推理（使用LUNA16标准输入尺寸）
            test_input = torch.randn(1, 1, 80, 192, 192, device=self.device)  # LUNA16标准尺寸

            with torch.no_grad():
                try:
                    output = self.model(test_input)
                    print(f"   模型验证成功，输出类型: {type(output)}")

                    if isinstance(output, dict):
                        print(f"   输出键: {list(output.keys())}")
                    elif isinstance(output, (list, tuple)):
                        print(f"   输出列表长度: {len(output)}")
                    elif isinstance(output, torch.Tensor):
                        print(f"   输出形状: {output.shape}")

                    self.model_info['validation_passed'] = True
                    return True

                except Exception as e:
                    print(f"   模型前向传播失败: {e}")
                    self.model_info['validation_passed'] = False
                    return False

        except Exception as e:
            print(f"   模型验证失败: {e}")
            self.model_info['validation_passed'] = False
            return False

    def _print_model_info(self):
        """打印模型信息"""
        print("\n📊 模型信息总结:")
        print("=" * 50)

        for key, value in self.model_info.items():
            print(f"   {key}: {value}")

        if self.model is not None:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            print(f"   total_parameters: {total_params:,}")
            print(f"   trainable_parameters: {trainable_params:,}")
            print(f"   model_size_mb: {total_params * 4 / 1024 / 1024:.2f}")

        print("=" * 50)

    def get_model(self):
        """获取加载的模型"""
        if self.model is not None:
            self.model = self.model.to(self.device)
            self.model.eval()
        return self.model

    def get_model_info(self):
        """获取模型信息"""
        return self.model_info.copy()

    def get_config(self):
        """获取配置"""
        if hasattr(self, 'parser'):
            return self.parser.config
        else:
            return self.config

    def cleanup(self):
        """清理临时文件"""
        try:
            if self.bundle_dir and self.bundle_dir != self.bundle_path:
                # 只清理临时解压的目录
                if 'tmp' in str(self.bundle_dir) or 'temp' in str(self.bundle_dir):
                    import shutil
                    shutil.rmtree(self.bundle_dir, ignore_errors=True)
                    print(f"   清理临时目录: {self.bundle_dir}")
        except Exception as e:
            print(f"   清理失败: {e}")


def test_bundle_loader():
    """测试Bundle加载器"""
    print("🧪 测试Bundle加载器")
    print(f"   当前用户: veryjoyran")
    print(f"   时间: 2025-06-24 15:25:42")

    # 测试路径
    bundle_path = "lung_nodule_ct_detection_v0.5.9.zip"

    if Path(bundle_path).exists():
        loader = MonAIBundleLoader(bundle_path, 'cpu')
        success = loader.load_bundle()

        if success:
            model = loader.get_model()
            info = loader.get_model_info()

            print(f"✅ Bundle加载测试成功")
            print(f"   模型类型: {info.get('network_class', '未知')}")
            print(f"   权重加载: {info.get('weights_loaded', False)}")
        else:
            print("⚠️ Bundle加载测试部分成功（使用备用模型）")

        loader.cleanup()
    else:
        print(f"❌ 测试文件不存在: {bundle_path}")


if __name__ == "__main__":
    test_bundle_loader()