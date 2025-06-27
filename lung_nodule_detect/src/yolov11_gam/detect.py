import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
import os
from ultralytics import YOLO


class YOLODetector:
    def __init__(self, weights='yolo11m.pt', conf_threshold=0.25, iou_threshold=0.45, device=''):
        # 修改设备选择逻辑
        self.device = self.get_available_device(device)
        print(f"使用设备: {self.device}")

        # 设置阈值
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # 加载模型
        self.model = self.load_model(weights)

        # 类别名称映射
        self.class_names = {0: '肺结节'}

        # 加载字体
        self.font = self.load_font()

    def get_available_device(self, preferred_device=''):
        """智能选择可用设备"""
        try:
            # 检查CUDA是否可用
            if torch.cuda.is_available():
                # 检查具体的GPU设备
                if preferred_device and preferred_device != 'cpu':
                    try:
                        # 尝试使用指定的GPU设备
                        device_num = int(preferred_device) if preferred_device.isdigit() else 0
                        if device_num < torch.cuda.device_count():
                            return f'cuda:{device_num}'
                        else:
                            print(f"GPU设备 {device_num} 不存在，回退到CPU")
                            return 'cpu'
                    except:
                        print("GPU设备配置错误，使用CPU")
                        return 'cpu'
                else:
                    return 'cuda:0'
            else:
                print("CUDA不可用，使用CPU")
                return 'cpu'
        except Exception as e:
            print(f"设备选择出错: {e}，使用CPU")
            return 'cpu'

    def load_font(self):
        """加载字体"""
        try:
            font_paths = [
                os.path.join(os.path.dirname(__file__), 'static/fonts/simhei.ttf'),
                'C:/Windows/Fonts/simhei.ttf',
                'C:/Windows/Fonts/msyh.ttf',
                '/System/Library/Fonts/STHeiti Light.ttc',
                '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'
            ]

            for font_path in font_paths:
                if os.path.exists(font_path):
                    return ImageFont.truetype(font_path, 16)

            return ImageFont.load_default()
        except Exception as e:
            print(f"字体加载失败: {e}")
            return ImageFont.load_default()

    def load_model(self, weights_path):
        """加载模型"""
        try:
            # 检查权重文件
            if not weights_path.startswith('yolo') and not os.path.exists(weights_path):
                print(f"权重文件不存在: {weights_path}，使用官方模型")
                weights_path = 'yolo11m.pt'

            # 创建模型
            model = YOLO(weights_path)

            # 显式设置设备
            model.to(self.device)

            # 设置参数
            model.conf = self.conf_threshold
            model.iou = self.iou_threshold

            print(f"模型成功加载: {weights_path}")
            print(f"模型设备: {model.device}")

            return model

        except Exception as e:
            print(f"模型加载失败: {e}")
            # 回退策略
            try:
                print("尝试加载YOLOv11n模型...")
                model = YOLO('yolo11n.pt')
                model.to(self.device)
                model.conf = self.conf_threshold
                model.iou = self.iou_threshold
                return model
            except Exception as e2:
                print(f"备用模型也加载失败: {e2}")
                raise e2

    def detect(self, image_path):
        """检测方法"""
        try:
            # 加载图像
            if isinstance(image_path, str):
                image = Image.open(image_path).convert('RGB')
            else:
                image = image_path.convert('RGB')

            img_width, img_height = image.size
            print(f"图像尺寸: {img_width}x{img_height}")

            # 执行检测 - 显式指定设备
            print(f"开始检测，使用设备: {self.device}")
            results = self.model(image, device=self.device, verbose=False)

            # 绘制结果
            img_result = image.copy()
            draw = ImageDraw.Draw(img_result)
            detections = []

            # 处理检测结果
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes
                    print(f"检测到 {len(boxes)} 个目标")

                    for box in boxes:
                        # 获取边界框坐标
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        xmin, ymin, xmax, ymax = int(x1), int(y1), int(x2), int(y2)

                        # 获取类别ID和置信度
                        cls_id = int(box.cls[0].item())
                        confidence = float(box.conf[0].item())

                        # 获取类别名称
                        if hasattr(self.model, 'names') and cls_id in self.model.names:
                            class_name = self.model.names[cls_id]
                        else:
                            class_name = self.class_names.get(cls_id, f"目标_{cls_id}")

                        # 绘制边界框
                        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=3)

                        # 绘制标签
                        label = f"{class_name} {confidence:.2f}"

                        # 文本背景和位置
                        try:
                            if hasattr(draw, 'textbbox'):
                                bbox = draw.textbbox((0, 0), label, font=self.font)
                                text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                            else:
                                text_w, text_h = draw.textsize(label, font=self.font)
                        except:
                            text_w, text_h = len(label) * 8, 16

                        label_x = max(0, min(xmin, img_width - text_w - 4))
                        label_y = max(0, ymin - text_h - 4) if ymin - text_h - 4 >= 0 else ymin + 4

                        # 绘制标签背景和文本
                        draw.rectangle(
                            [(label_x, label_y), (label_x + text_w + 4, label_y + text_h + 4)],
                            fill="red"
                        )
                        draw.text((label_x + 2, label_y + 2), label, fill="white", font=self.font)

                        # 添加到检测结果
                        detections.append({
                            'class': class_name,
                            'confidence': f"{confidence:.2f}",
                            'xmin': xmin,
                            'ymin': ymin,
                            'xmax': xmax,
                            'ymax': ymax
                        })

            print(f"检测完成，发现 {len(detections)} 个目标")
            return img_result, detections

        except Exception as e:
            print(f"检测过程出错: {e}")
            import traceback
            traceback.print_exc()
            # 返回原图和空结果
            return image, []