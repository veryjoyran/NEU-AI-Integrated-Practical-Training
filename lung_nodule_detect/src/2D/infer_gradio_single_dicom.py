
import gradio as gr
from pathlib import Path
from datetime import datetime

from matplotlib import pyplot as plt

# 导入改进的检测器
from single_dicom_detector import ImprovedSingleDicomDetector

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ImprovedDicomGradioInterface:
    """改进版DICOM检测Gradio界面"""

    def __init__(self):
        self.detector = None

    def load_bundle(self, bundle_file):
        """加载Bundle"""
        try:
            if bundle_file is None:
                return "❌ 请上传MonAI Bundle文件", "未加载"

            bundle_path = bundle_file.name

            self.detector = ImprovedSingleDicomDetector()
            success = self.detector.load_bundle(bundle_path)

            if success:
                model_info = self.detector.model_info

                info_text = f"""
✅ 改进版Bundle加载成功!

📁 Bundle文件: {Path(bundle_path).name}
🏗️ 模型类型: {model_info.get('type', 'Unknown')}
📅 加载时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
🖥️ 运行设备: {self.detector.device}

🎯 改进版特性:
  • 复用YOLO成功的预处理模块
  • 智能切片选择算法
  • CLAHE对比度增强
  • 多版本预处理测试
  • 医学图像专用窗宽窗位

🚀 请上传DICOM文件开始改进版检测!
"""

                return info_text, "改进版Bundle已加载"
            else:
                return "⚠️ Bundle加载部分成功，使用默认配置", "部分加载"

        except Exception as e:
            error_msg = f"❌ Bundle加载失败: {str(e)}"
            return error_msg, "加载失败"

    def process_improved_dicom(self, dicom_file, window_center, window_width, test_all_versions):
        """改进版DICOM处理"""
        try:
            if self.detector is None:
                return None, "❌ 请先加载MonAI Bundle"

            if dicom_file is None:
                return None, "❌ 请上传DICOM文件"

            dicom_path = dicom_file.name
            print(f"🔄 改进版处理DICOM: {dicom_path}")

            # 使用改进的检测方法
            result = self.detector.detect_with_improved_preprocessing(
                dicom_path,
                window_center=int(window_center),
                window_width=int(window_width),
                test_all_versions=test_all_versions
            )

            if result is None:
                return None, "❌ 改进版检测失败，请检查DICOM文件格式和Bundle兼容性"

            # 生成可视化
            fig = self.detector.visualize_improved_result(result)

            # 生成报告
            report = self.detector.generate_improved_report(result, dicom_path)

            return fig, report

        except Exception as e:
            error_msg = f"❌ 改进版DICOM处理失败: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return None, error_msg

    def create_interface(self):
        """创建改进版界面"""

        with gr.Blocks(title="改进版DICOM肺结节检测", theme=gr.themes.Soft()) as interface:

            gr.HTML("""
            <h1 style='text-align: center; color: #2c3e50;'>
                🫁 改进版DICOM肺结节检测系统 
                <span style='background-color: #e74c3c; color: white; padding: 4px 8px; border-radius: 4px; font-size: 14px;'>
                    IMPROVED v2.0.0
                </span>
            </h1>
            """)

            gr.Markdown("""
            ### 🎯 改进版系统特点
            
            - ✅ **复用YOLO成功模块**: 集成在YOLO中验证有效的预处理算法
            - ✅ **智能切片选择**: 自动选择信息量最大的切片
            - ✅ **CLAHE对比度增强**: 医学图像专用的对比度优化
            - ✅ **多版本预处理测试**: 自动测试多种预处理方法找出最佳配置
            - ✅ **医学专用窗宽窗位**: 针对肺部结构优化的显示参数
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("<h3>🤖 1. 加载检测模型</h3>")

                    bundle_file = gr.File(
                        label="上传MonAI Bundle文件 (.zip)",
                        file_types=[".zip"],
                        file_count="single"
                    )

                    load_bundle_btn = gr.Button("🚀 加载改进版Bundle", variant="primary")

                    bundle_status = gr.Textbox(
                        label="Bundle状态",
                        value="未加载",
                        interactive=False,
                        lines=1
                    )

                    bundle_info = gr.Textbox(
                        label="改进版Bundle信息",
                        lines=10,
                        interactive=False,
                        value="🔄 请上传Bundle文件启动改进版检测..."
                    )

                with gr.Column(scale=1):
                    gr.HTML("<h3>📄 2. 上传DICOM文件</h3>")

                    dicom_file = gr.File(
                        label="上传单张DICOM文件 (.dcm)",
                        file_types=[".dcm"],
                        file_count="single"
                    )

                    gr.HTML("<h4>⚙️ 改进版检测参数</h4>")

                    with gr.Row():
                        window_center = gr.Slider(
                            label="窗位 (Window Center)",
                            minimum=-200,
                            maximum=200,
                            value=50,
                            step=10,
                            info="肺部推荐: 50, 骨骼: 300"
                        )

                        window_width = gr.Slider(
                            label="窗宽 (Window Width)",
                            minimum=100,
                            maximum=1000,
                            value=350,
                            step=50,
                            info="肺部推荐: 350, 骨骼: 1500"
                        )

                    test_all_versions = gr.Checkbox(
                        label="🧪 启用多版本预处理测试",
                        value=True,
                        info="自动测试多种预处理方法，找出检测效果最佳的配置"
                    )

                    detect_btn = gr.Button("🔍 开始改进版检测", variant="primary", size="lg")

                    gr.Markdown("""
                    **💡 改进版使用提示:**
                    - **多版本测试**: 会自动尝试多种预处理配置
                    - **窗宽窗位**: 已针对肺部结构进行优化
                    - **CLAHE增强**: 自动应用医学图像对比度增强
                    - **智能切片**: 自动选择信息量最大的切片
                    """)

            gr.HTML("<h3>📊 3. 改进版检测结果</h3>")

            with gr.Row():
                result_visualization = gr.Plot(
                    label="改进版检测结果可视化",
                    show_label=True
                )

                improved_report = gr.Textbox(
                    label="改进版检测报告",
                    lines=25,
                    interactive=False,
                    value="""🔄 请先加载Bundle并上传DICOM文件...

📋 改进版检测流程:
1️⃣ 上传MonAI Bundle文件
2️⃣ 上传单张DICOM文件 (.dcm)  
3️⃣ 调整窗宽窗位参数 (可选)
4️⃣ 启用多版本测试 (推荐)
5️⃣ 点击开始检测
6️⃣ 查看详细分析结果

🎯 改进版优势:
• 复用YOLO验证成功的预处理算法
• 智能切片选择 - 自动找到最佳切片
• CLAHE对比度增强 - 提升图像质量
• 多版本测试 - 找出最适合的预处理方法
• 医学专用窗宽窗位 - 优化肺部结构显示

💡 如果检测不到结节:
• 系统会自动测试多种预处理方法
• 提供详细的失败分析
• 给出针对性的改进建议"""
                )

            # 事件绑定
            load_bundle_btn.click(
                fn=self.load_bundle,
                inputs=[bundle_file],
                outputs=[bundle_info, bundle_status]
            )

            detect_btn.click(
                fn=self.process_improved_dicom,
                inputs=[dicom_file, window_center, window_width, test_all_versions],
                outputs=[result_visualization, improved_report],
                show_progress=True
            )

            gr.Markdown("""
            ---
            ### 📋 改进版使用指南
            
            #### 🔧 YOLO成功模块复用:
            
            本改进版系统复用了在YOLO中验证成功的以下模块：
            
            **1. 智能切片选择算法**
            ```
            • 基于标准差的质量评估
            • 动态范围分析
            • 组合评分机制
            ```
            
            **2. CLAHE对比度增强**
            ```
            • 自适应直方图均衡化
            • 医学图像专用参数
            • 局部对比度优化
            ```
            
            **3. 医学专用窗宽窗位**
            ```
            • 肺部结构优化显示
            • HU值范围标准化
            • 多种显示模式支持
            ```
            
            **4. 多版本预处理测试**
            ```
            • MonAI标准归一化 [0, 1]
            • Z-score标准化 (零均值单位方差)
            • HU值重映射 [-1000, 400]
            • 直接uint8处理
            ```
            
            #### 🎯 检测失败诊断:
            
            如果所有Bundle版本都检测不到结节，改进版系统会：
            
            1. **自动测试多种预处理方法**
            2. **提供详细的失败分析报告**
            3. **给出针对性的改进建议**
            4. **显示每种方法的测试结果**
            
            #### 💡 参数调优建议:
            
            **窗宽窗位设置:**
            - 肺部: 窗位50, 窗宽350 (默认)
            - 纵隔: 窗位40, 窗宽400
            - 骨骼: 窗位300, 窗宽1500
            
            **多版本测试:**
            - 推荐启用，可以找出最适合当前Bundle的预处理方法
            - 如果单版本失败，多版本测试可能会发现有效的配置
            
            #### 🔍 结果解读:
            
            **多版本测试结果:**
            - ✅ 绿色: 该预处理方法检测成功
            - ❌ 红色: 该预处理方法未检测到结节
            - 系统会自动推荐最佳的预处理方法
            
            **检测质量评估:**
            - 置信度 >0.7: 高质量检测
            - 置信度 0.3-0.7: 中等质量检测  
            - 置信度 <0.3: 低质量检测，需要验证
            
            ---
            
            **版本**: 改进版 v2.0.0 | **用户**: veryjoyran | **时间**: 2025-06-24 03:38:22
            
            **关键改进**: 集成YOLO成功验证的预处理模块，提供多版本自动测试功能
            """)

        return interface


def main():
    """主函数"""
    print("🚀 启动改进版DICOM肺结节检测界面")
    print(f"👤 用户: veryjoyran")
    print(f"📅 时间: 2025-06-24 03:38:22")
    print("🎯 改进版: 集成YOLO成功模块 + 多版本预处理测试")
    print("=" * 70)

    try:
        app = ImprovedDicomGradioInterface()
        interface = app.create_interface()

        print("✅ 改进版界面创建完成")
        print("📌 主要改进:")
        print("   • 复用YOLO中验证有效的预处理模块")
        print("   • 智能切片选择和CLAHE对比度增强")
        print("   • 多版本预处理自动测试")
        print("   • 详细的失败诊断和改进建议")

        interface.launch(
            server_name="127.0.0.1",
            server_port=7863,  # 避免端口冲突
            debug=True,
            show_error=True,
            inbrowser=True,
            share=False
        )

    except Exception as e:
        print(f"❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()