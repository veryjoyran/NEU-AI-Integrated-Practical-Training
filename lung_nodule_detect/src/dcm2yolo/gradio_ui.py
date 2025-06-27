import gradio as gr
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import tempfile
import os
import torch
from ultralytics import YOLO
import pydicom
from PIL import Image
import json
import time

# 导入我们的预处理模块
from dicom_processor_v3 import DICOMProcessorV3
from lung_segmentation_preprocessor import LungSegmentationPreprocessor


class LungNoduleDetectionUI:
    """肺结节检测UI界面"""

    def __init__(self):
        self.dicom_processor = DICOMProcessorV3()
        self.lung_processor = LungSegmentationPreprocessor()
        self.current_model = None
        self.current_model_path = None
        self.processing_cache = {}

        # 设置matplotlib中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

    def load_model(self, model_path):
        """加载YOLOv11模型"""
        try:
            if not model_path or not os.path.exists(model_path):
                return "❌ 模型文件不存在，请检查路径", None

            # 加载模型
            model = YOLO(model_path)
            self.current_model = model
            self.current_model_path = model_path

            # 获取模型信息
            model_info = {
                "model_path": model_path,
                "model_name": Path(model_path).name,
                "device": "GPU" if torch.cuda.is_available() else "CPU",
                "classes": getattr(model.model, 'names', {0: 'nodule'}),
                "input_size": 640  # YOLOv11默认输入尺寸
            }

            info_text = f"""
✅ 模型加载成功！

📁 模型路径: {model_info['model_name']}
🖥️ 运行设备: {model_info['device']}
🎯 检测类别: {list(model_info['classes'].values())}
📐 输入尺寸: {model_info['input_size']}x{model_info['input_size']}
            """

            return info_text, model_info

        except Exception as e:
            return f"❌ 模型加载失败: {str(e)}", None

    def process_dicom(self, dicom_file, enable_lung_segmentation=True, target_size=640):
        """处理DICOM文件"""
        try:
            if dicom_file is None:
                return None, None, None, "❌ 请上传DICOM文件"

            # 读取DICOM文件
            dcm_path = Path(dicom_file.name)

            print(f"🔄 Processing DICOM: {dcm_path.name}")

            # Step 1: DICOM基础处理
            result = self.dicom_processor.process_dicom_image(dcm_path, target_size=(target_size, target_size))

            if result[0] is None:
                return None, None, None, "❌ DICOM文件处理失败"

            processed_8bit_image = result[0]
            original_shape = result[1]
            modality = result[2]
            strategy = result[3]

            # Step 2: 可选的肺分割处理
            lung_segmented_image = None
            lung_info = None

            if enable_lung_segmentation and modality == "CT":
                print("🫁 Applying lung segmentation...")
                lung_result = self.lung_processor.process_8bit_image(processed_8bit_image, dcm_path.stem)

                if lung_result['success']:
                    lung_segmented_image = lung_result['processed_image']
                    lung_info = {
                        'lung_mask': lung_result['lung_mask'],
                        'lung_bbox': lung_result['lung_bbox'],
                        'left_lung_mask': lung_result['left_lung_mask'],
                        'right_lung_mask': lung_result['right_lung_mask']
                    }
                    print("✅ Lung segmentation successful")
                else:
                    print("⚠️ Lung segmentation failed, using original processed image")
                    lung_segmented_image = processed_8bit_image
            else:
                lung_segmented_image = processed_8bit_image

            # 创建对比显示图像
            comparison_image = self.create_preprocessing_comparison(
                processed_8bit_image, lung_segmented_image, lung_info,
                modality, strategy, enable_lung_segmentation
            )

            # 缓存处理结果
            cache_key = f"{dcm_path.name}_{enable_lung_segmentation}_{target_size}"
            self.processing_cache[cache_key] = {
                'original_8bit': processed_8bit_image,
                'lung_segmented': lung_segmented_image,
                'lung_info': lung_info,
                'modality': modality,
                'strategy': strategy,
                'original_shape': original_shape,
                'target_size': target_size
            }

            # 处理信息
            processing_info = f"""
🔍 DICOM处理信息:
  📁 文件名: {dcm_path.name}
  🖼️ 图像模态: {modality}
  🛠️ 处理策略: {strategy}
  📐 原始尺寸: {original_shape}
  📐 目标尺寸: {target_size}x{target_size}
  🫁 肺分割: {'启用' if enable_lung_segmentation else '禁用'}
"""

            if lung_info and lung_info.get('lung_mask') is not None:
                lung_area = np.sum(lung_info['lung_mask'])
                total_area = lung_info['lung_mask'].size
                lung_percentage = (lung_area / total_area) * 100
                processing_info += f"  📊 肺区域占比: {lung_percentage:.1f}%\n"

            return comparison_image, lung_segmented_image, cache_key, processing_info

        except Exception as e:
            return None, None, None, f"❌ 处理错误: {str(e)}"

    def create_preprocessing_comparison(self, original_8bit, lung_segmented, lung_info,
                                        modality, strategy, enable_lung_segmentation):
        """创建预处理对比图像"""
        try:
            if enable_lung_segmentation and lung_segmented is not None:
                # 三图对比：原图、分割结果、轮廓叠加
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                fig.suptitle(f'DICOM预处理结果 - {modality} ({strategy})', fontsize=16, fontweight='bold')

                # 原图
                axes[0].imshow(original_8bit, cmap='gray')
                axes[0].set_title('原始8位图像', fontweight='bold')
                axes[0].axis('off')

                # 肺分割结果
                axes[1].imshow(lung_segmented, cmap='gray')
                axes[1].set_title('肺分割结果', fontweight='bold')
                axes[1].axis('off')

                # 轮廓叠加
                axes[2].imshow(original_8bit, cmap='gray')
                if lung_info and lung_info.get('lung_mask') is not None:
                    from skimage import segmentation
                    boundaries = segmentation.find_boundaries(lung_info['lung_mask'], mode='thick')
                    boundary_coords = np.where(boundaries)
                    if len(boundary_coords[0]) > 0:
                        axes[2].scatter(boundary_coords[1], boundary_coords[0],
                                        c='red', s=0.3, alpha=0.8, linewidths=0)

                axes[2].set_title('轮廓叠加', fontweight='bold')
                axes[2].axis('off')

            else:
                # 单图显示
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                fig.suptitle(f'DICOM处理结果 - {modality} ({strategy})', fontsize=16, fontweight='bold')

                ax.imshow(original_8bit, cmap='gray')
                ax.set_title('处理后图像', fontweight='bold')
                ax.axis('off')

            plt.tight_layout()

            # 保存到临时文件
            temp_path = tempfile.mktemp(suffix='.png')
            plt.savefig(temp_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()

            return temp_path

        except Exception as e:
            print(f"创建对比图像失败: {e}")
            return None

    def detect_nodules(self, cache_key, confidence_threshold=0.25, iou_threshold=0.45):
        """使用YOLOv11进行肺结节检测"""
        try:
            if self.current_model is None:
                return None, "❌ 请先加载模型"

            if cache_key not in self.processing_cache:
                return None, "❌ 请先处理DICOM图像"

            cached_data = self.processing_cache[cache_key]
            input_image = cached_data['lung_segmented']  # 使用肺分割后的图像

            print(f"🔍 Running YOLOv11 inference...")
            print(f"   Confidence threshold: {confidence_threshold}")
            print(f"   IoU threshold: {iou_threshold}")

            # 转换为RGB格式（YOLOv11需要3通道输入）
            if len(input_image.shape) == 2:
                input_rgb = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
            else:
                input_rgb = input_image

            # 进行推理
            results = self.current_model(
                input_rgb,
                conf=confidence_threshold,
                iou=iou_threshold,
                verbose=False
            )

            # 解析检测结果
            detections = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes

                for i in range(len(boxes)):
                    # 获取边界框坐标 (xyxy格式)
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    confidence = boxes.conf[i].cpu().numpy()
                    class_id = int(boxes.cls[i].cpu().numpy())

                    # 获取类别名称
                    class_name = self.current_model.names.get(class_id, f'class_{class_id}')

                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(confidence),
                        'class_id': class_id,
                        'class_name': class_name
                    })

            # 创建检测结果可视化
            result_image = self.create_detection_visualization(
                input_image, detections, cached_data
            )

            # 创建检测信息
            detection_info = self.create_detection_info(detections, cached_data)

            return result_image, detection_info

        except Exception as e:
            return None, f"❌ 检测错误: {str(e)}"

    def create_detection_visualization(self, input_image, detections, cached_data):
        """创建检测结果可视化"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle('YOLOv11肺结节检测结果', fontsize=16, fontweight='bold')

            # 左图：输入图像
            axes[0].imshow(input_image, cmap='gray')
            axes[0].set_title(f'输入图像 ({cached_data["modality"]})', fontweight='bold')
            axes[0].axis('off')

            # 右图：检测结果
            axes[1].imshow(input_image, cmap='gray')
            axes[1].set_title(f'检测结果 (共{len(detections)}个结节)', fontweight='bold')

            # 绘制检测框
            colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
            for i, detection in enumerate(detections):
                x1, y1, x2, y2 = detection['bbox']
                confidence = detection['confidence']
                class_name = detection['class_name']

                # 选择颜色
                color = colors[i % len(colors)]

                # 绘制边界框
                rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor=color, facecolor='none')
                axes[1].add_patch(rect)

                # 添加标签
                label = f'{class_name}: {confidence:.2f}'
                axes[1].text(x1, y1 - 5, label, color=color, fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))

            axes[1].axis('off')

            plt.tight_layout()

            # 保存到临时文件
            temp_path = tempfile.mktemp(suffix='.png')
            plt.savefig(temp_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()

            return temp_path

        except Exception as e:
            print(f"创建检测可视化失败: {e}")
            return None

    def create_detection_info(self, detections, cached_data):
        """创建检测信息文本"""
        info_text = f"""
🎯 YOLOv11检测结果:
  📊 检测到结节数量: {len(detections)}
  🖼️ 图像尺寸: {cached_data['target_size']}x{cached_data['target_size']}
  🛠️ 预处理策略: {cached_data['strategy']}

📋 检测详情:
"""

        if len(detections) == 0:
            info_text += "  ❌ 未检测到肺结节\n"
        else:
            for i, detection in enumerate(detections, 1):
                x1, y1, x2, y2 = detection['bbox']
                confidence = detection['confidence']
                class_name = detection['class_name']

                # 计算中心点和尺寸
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1

                info_text += f"""
  🔍 结节 {i}:
    • 类别: {class_name}
    • 置信度: {confidence:.3f}
    • 位置: ({center_x:.1f}, {center_y:.1f})
    • 尺寸: {width:.1f} × {height:.1f} 像素
    • 边界框: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]
"""

        return info_text

    def create_gradio_interface(self):
        """创建Gradio界面"""

        with gr.Blocks(title="肺结节检测系统", theme=gr.themes.Soft()) as interface:
            gr.Markdown("""
            # 🫁 肺结节检测系统

            基于YOLOv11的DICOM图像肺结节检测系统，支持DICOM预处理和肺分割功能。

            ## 📋 使用步骤：
            1. **加载模型**: 上传训练好的YOLOv11权重文件 (.pt)
            2. **上传DICOM**: 选择要检测的DICOM文件 (.dcm)
            3. **预处理设置**: 配置肺分割和检测参数
            4. **开始检测**: 查看预处理结果和检测结果
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🤖 模型配置")

                    model_file = gr.File(
                        label="上传YOLOv11模型文件 (.pt)",
                        file_types=[".pt"],
                        file_count="single"
                    )

                    load_model_btn = gr.Button("加载模型", variant="primary", size="sm")
                    model_info = gr.Textbox(
                        label="模型信息",
                        lines=8,
                        interactive=False,
                        value="🔄 请上传并加载模型..."
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### 📁 DICOM文件")

                    dicom_file = gr.File(
                        label="上传DICOM文件 (.dcm)",
                        file_types=[".dcm"],
                        file_count="single"
                    )

                    with gr.Row():
                        enable_lung_seg = gr.Checkbox(
                            label="启用肺分割",
                            value=True
                        )
                        target_size = gr.Slider(
                            label="目标图像尺寸",
                            minimum=256,
                            maximum=1024,
                            value=640,
                            step=64
                        )

                    process_btn = gr.Button("处理DICOM", variant="secondary", size="sm")
                    processing_info = gr.Textbox(
                        label="处理信息",
                        lines=8,
                        interactive=False,
                        value="🔄 请上传DICOM文件..."
                    )

            gr.Markdown("### 🖼️ 预处理结果")
            preprocessing_result = gr.Image(
                label="DICOM预处理对比",
                show_label=True,
                interactive=False
            )

            gr.Markdown("### 🎯 检测配置与结果")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### ⚙️ 检测参数")

                    confidence_threshold = gr.Slider(
                        label="置信度阈值",
                        minimum=0.1,
                        maximum=0.9,
                        value=0.25,
                        step=0.05,
                        info="检测结果的最小置信度"
                    )

                    iou_threshold = gr.Slider(
                        label="IoU阈值",
                        minimum=0.1,
                        maximum=0.9,
                        value=0.45,
                        step=0.05,
                        info="非极大值抑制的IoU阈值"
                    )

                    detect_btn = gr.Button("开始检测", variant="primary", size="lg")

                    detection_info = gr.Textbox(
                        label="检测结果信息",
                        lines=15,
                        interactive=False,
                        value="🔄 请先处理DICOM并加载模型..."
                    )

                with gr.Column(scale=2):
                    detection_result = gr.Image(
                        label="检测结果可视化",
                        show_label=True,
                        interactive=False
                    )

            # 隐藏状态变量
            model_state = gr.State(None)
            cache_key_state = gr.State(None)

            # 事件绑定
            load_model_btn.click(
                fn=self.load_model,
                inputs=[model_file],
                outputs=[model_info, model_state]
            )

            process_btn.click(
                fn=self.process_dicom,
                inputs=[dicom_file, enable_lung_seg, target_size],
                outputs=[preprocessing_result, gr.State(), cache_key_state, processing_info]
            )

            detect_btn.click(
                fn=self.detect_nodules,
                inputs=[cache_key_state, confidence_threshold, iou_threshold],
                outputs=[detection_result, detection_info]
            )

            # 添加示例和说明
            gr.Markdown("""
            ## 📖 使用说明

            ### 🔧 预处理功能
            - **DICOM处理**: 自动进行HU值转换、窗宽窗位调整等
            - **肺分割**: 基于形态学操作的肺区域分割，去除肺外干扰
            - **对比显示**: 显示原图、分割结果和轮廓叠加

            ### 🎯 检测功能  
            - **YOLOv11推理**: 使用训练好的模型进行肺结节检测
            - **参数调整**: 可调整置信度和IoU阈值
            - **结果可视化**: 显示检测框和置信度分数

            ### 📊 输出信息
            - **预处理统计**: 肺区域占比、处理策略等
            - **检测结果**: 结节数量、位置、置信度等详细信息

            ---
            💡 **提示**: 建议先在小批量数据上测试，确认预处理和检测效果后再批量处理。
            """)

        return interface


def main():
    """主函数"""
    print("🚀 启动肺结节检测UI界面...")

    # 创建UI实例
    ui = LungNoduleDetectionUI()

    # 创建界面
    interface = ui.create_gradio_interface()

    # 启动界面
    interface.launch(
        server_name="127.0.0.1",  # 允许外部访问
        server_port=7860,  # 默认端口
        share=False,  # 不创建公共链接
        debug=True,  # 启用调试模式
        show_error=True  # 显示错误信息
    )


if __name__ == "__main__":
    main()