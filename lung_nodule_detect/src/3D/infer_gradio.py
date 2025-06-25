"""
3D肺结节检测Gradio界面 - LIDC兼容中文版
Author: veryjoyran
Date: 2025-06-24 15:31:04
"""

import gradio as gr
import shutil
import tempfile
from pathlib import Path
import zipfile
import os
from datetime import datetime
import matplotlib.pyplot as plt

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 导入3D检测器
try:
    from inference_only_3d_detector import Pure3DDetector
    print("✅ 成功导入3D检测器")
except ImportError as e:
    print(f"❌ 导入3D检测器失败: {e}")
    raise

class LUNA16GradioInterface:
    """LUNA16/LIDC兼容的3D检测Gradio界面"""

    def __init__(self):
        self.detector = None
        self.current_bundle_path = None

        print(f"🚀 初始化LUNA16/LIDC Gradio界面")
        print(f"   当前用户: veryjoyran")
        print(f"   时间: 2025-06-24 15:31:04")

    def load_bundle_3d(self, bundle_file):
        """加载3D Bundle"""
        try:
            if bundle_file is None:
                return "❌ 请上传MonAI Bundle文件", "未加载"

            bundle_path = bundle_file.name
            self.current_bundle_path = bundle_path

            print(f"🔄 加载3D Bundle: {bundle_path}")

            # 初始化3D检测器
            self.detector = Pure3DDetector()
            success = self.detector.load_bundle(bundle_path)

            if success:
                model_info = self.detector.model_info

                info_text = f"""
✅ 3D LUNA16 Bundle加载成功！

📁 Bundle文件: {Path(bundle_path).name}
🏗️ 模型类型: {model_info.get('network_class', '未知')}
📅 加载时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
🖥️ 运行设备: {self.detector.device}
⚙️ 配置解析器: {model_info.get('config_parser_used', False)}
🎯 权重已加载: {model_info.get('weights_loaded', False)}
📊 加载成功率: {model_info.get('load_ratio', 0):.2%}

🎯 3D LUNA16特性:
  • 3D体积处理（非逐片处理）
  • LUNA16标准预处理流程
  • 体素间距: 0.703125 x 0.703125 x 1.25 mm
  • 模型输入尺寸: 192 x 192 x 80
  • LIDC数据集兼容
  • 多版本预处理测试

🔬 LIDC数据集兼容性:
  • ✅ 与LUNA16相同的源数据 (LIDC-IDRI)
  • ✅ 兼容的预处理流程
  • ✅ 处理可变的LIDC扫描协议
  • ✅ 考虑LIDC中的<3mm结节
  • ✅ 3D上下文分析提高准确性

🚀 准备开始3D LIDC检测！
"""

                return info_text, "3D Bundle已加载"
            else:
                return "⚠️ Bundle部分加载成功，使用备用模型", "部分加载"

        except Exception as e:
            error_msg = f"❌ 3D Bundle加载失败: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return error_msg, "加载失败"

    def process_dicom_zip_3d(self, zip_file, test_all_versions):
        """处理DICOM ZIP文件 - 3D模式"""
        try:
            if self.detector is None:
                return None, "❌ 请先加载MonAI Bundle"

            if zip_file is None:
                return None, "❌ 请上传DICOM ZIP文件"

            print(f"🔄 处理DICOM ZIP (3D LIDC模式): {zip_file.name}")

            # 解压DICOM文件
            temp_dir = Path(tempfile.mkdtemp(prefix="lidc_dicom_"))
            try:
                with zipfile.ZipFile(zip_file.name, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)

                # 查找DICOM文件
                dicom_files = []
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        if file.lower().endswith(('.dcm', '.dicom')):
                            dicom_files.append(Path(root) / file)

                if not dicom_files:
                    return None, "❌ ZIP文件中未找到DICOM文件"

                print(f"   找到 {len(dicom_files)} 个DICOM文件")

                # 使用DICOM文件目录进行3D检测
                dicom_series_dir = dicom_files[0].parent if len(dicom_files) > 1 else dicom_files[0]

                return self._process_dicom_3d(dicom_series_dir, test_all_versions)

            finally:
                # 清理临时目录
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except:
                    pass

        except Exception as e:
            error_msg = f"❌ DICOM ZIP处理失败: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return None, error_msg

    def process_multiple_dicoms_3d(self, dicom_files, test_all_versions):
        """处理多个DICOM文件 - 3D模式"""
        try:
            if self.detector is None:
                return None, "❌ 请先加载MonAI Bundle"

            if not dicom_files:
                return None, "❌ 请上传DICOM文件"

            print(f"🔄 处理 {len(dicom_files)} 个DICOM文件 (3D LIDC模式)")

            # 创建临时目录并复制文件
            temp_dir = Path(tempfile.mkdtemp(prefix="lidc_series_"))
            try:
                for i, file in enumerate(dicom_files):
                    dest_path = temp_dir / f"slice_{i:04d}.dcm"
                    shutil.copy(file.name, dest_path)

                print(f"   文件已复制到: {temp_dir}")

                return self._process_dicom_3d(temp_dir, test_all_versions)

            finally:
                # 清理临时目录
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except:
                    pass

        except Exception as e:
            error_msg = f"❌ 多个DICOM文件处理失败: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return None, error_msg

    def _process_dicom_3d(self, dicom_path, test_all_versions):
        """3D DICOM处理核心函数"""
        try:
            print(f"🔍 开始3D LIDC检测...")
            print(f"   输入: {dicom_path}")
            print(f"   测试所有版本: {test_all_versions}")

            # 🔥 执行3D检测
            result = self.detector.detect_3d(
                dicom_path,
                test_all_versions=test_all_versions
            )

            if result is None:
                return None, """❌ 3D检测失败

💡 LIDC数据可能的问题:
• LIDC扫描协议可能与LUNA16训练数据不同
• 某些LIDC扫描具有不同的层厚
• LIDC中的结节<3mm可能检测不到（LUNA16训练于>3mm）
• 考虑检查DICOM文件完整性

🔧 故障排除步骤:
• 验证DICOM文件是有效的胸部CT扫描
• 检查文件是否包含肺部解剖结构
• 尝试不同的LIDC病例
• 考虑调整置信度阈值"""

            # 生成可视化
            fig = self.detector.visualize_3d_result(result)

            # 生成详细报告
            report = self.detector.generate_3d_report(result, dicom_path)

            return fig, report

        except Exception as e:
            error_msg = f"❌ 3D DICOM处理失败: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return None, error_msg

    def create_interface(self):
        """创建3D Gradio界面"""

        custom_css = """
        .main-title { 
            font-size: 24px; 
            font-weight: bold; 
            text-align: center; 
            margin-bottom: 20px; 
            color: #2c3e50; 
        }
        .section-title { 
            font-size: 18px; 
            font-weight: bold; 
            margin-top: 15px; 
            margin-bottom: 10px; 
            color: #34495e; 
        }
        .info-box { 
            background-color: #e8f6f3; 
            padding: 15px; 
            border-radius: 8px; 
            margin: 10px 0; 
            border-left: 4px solid #1abc9c; 
        }
        .warning-box { 
            background-color: #fdf2e9; 
            padding: 15px; 
            border-radius: 8px; 
            margin: 10px 0; 
            border-left: 4px solid #e67e22; 
        }
        .lidc-box { 
            background-color: #f0f8ff; 
            padding: 15px; 
            border-radius: 8px; 
            margin: 10px 0; 
            border-left: 4px solid #4169e1; 
        }
        .3d-badge { 
            background-color: #e74c3c; 
            color: white; 
            padding: 4px 8px; 
            border-radius: 4px; 
            font-size: 12px; 
        }
        .lidc-badge { 
            background-color: #4169e1; 
            color: white; 
            padding: 4px 8px; 
            border-radius: 4px; 
            font-size: 12px; 
        }
        """

        with gr.Blocks(title="3D LUNA16/LIDC肺结节检测", css=custom_css, theme=gr.themes.Soft()) as interface:

            gr.HTML("""
            <div class='main-title'>
                🫁 3D LUNA16/LIDC肺结节检测系统 
                <span class='3d-badge'>3D v5.0.0</span>
                <span class='lidc-badge'>LIDC兼容</span>
            </div>
            """)

            gr.Markdown("""
            <div class='info-box'>
            <b>🎯 3D LUNA16/LIDC检测系统特性:</b><br>
            • ✅ <b>真正的3D处理</b>: 完整的体积分析，而非逐片处理<br>
            • ✅ <b>LUNA16标准</b>: 与模型训练完全一致的预处理（0.703125mm间距，192×192×80）<br>
            • ✅ <b>LIDC兼容</b>: 处理原始LIDC-IDRI数据集变化<br>
            • ✅ <b>多版本测试</b>: 自动测试最优预处理方法<br>
            • ✅ <b>3D上下文</b>: 利用所有切片间的空间关系
            </div>
            """)

            gr.Markdown("""
            <div class='lidc-box'>
            <b>📊 LIDC数据集兼容性说明:</b><br>
            • <b>数据源</b>: LIDC-IDRI是LUNA16派生的原始数据集<br>
            • <b>差异</b>: LIDC包含所有结节（包括<3mm），LUNA16只过滤到>3mm<br>
            • <b>兼容性</b>: 模型在LUNA16子集上训练，但与LIDC源数据完全兼容<br>
            • <b>检测</b>: 可能检测到比LUNA16挑战更小的结节，请考虑置信度阈值<br>
            • <b>注释</b>: LIDC有4名放射科医师共识，LUNA16有处理的真值
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("<div class='section-title'>🤖 MonAI Bundle配置</div>")

                    bundle_file = gr.File(
                        label="上传MonAI Bundle文件 (.zip)",
                        file_types=[".zip"],
                        file_count="single"
                    )

                    gr.Markdown("""
                    <div class='info-box'>
                    <b>💡 3D Bundle要求:</b><br>
                    • lung_nodule_ct_detection_v0.5.x.zip (LUNA16训练)<br>
                    • RetinaNet或兼容的3D检测模型<br>
                    • 自动3D模型加载和验证<br>
                    • LIDC兼容预处理流程
                    </div>
                    """)

                    load_bundle_btn = gr.Button("🚀 加载3D Bundle", variant="primary", size="sm")

                    bundle_status = gr.Textbox(
                        label="Bundle状态",
                        value="未加载",
                        interactive=False,
                        lines=1
                    )

                    bundle_info = gr.Textbox(
                        label="3D Bundle信息",
                        lines=18,
                        interactive=False,
                        value="🔄 请上传MonAI Bundle文件开始3D LIDC检测..."
                    )

                with gr.Column(scale=1):
                    gr.HTML("<div class='section-title'>📁 LIDC DICOM数据上传</div>")

                    with gr.Tabs():
                        with gr.TabItem("🗂️ ZIP压缩包（推荐）"):
                            gr.Markdown("""
                            <div class='lidc-box'>
                            <b>📦 LIDC DICOM ZIP上传:</b><br>
                            • 上传完整的LIDC病例作为ZIP文件<br>
                            • 系统自动处理整个3D体积<br>
                            • 保持切片间的空间关系<br>
                            • 最适合3D结节检测准确性
                            </div>
                            """)

                            dicom_zip = gr.File(
                                label="上传LIDC DICOM序列ZIP",
                                file_types=[".zip"],
                                file_count="single"
                            )

                            process_zip_btn = gr.Button("🔍 3D处理ZIP", variant="primary", size="lg")

                        with gr.TabItem("📄 多个DICOM文件"):
                            gr.Markdown("""
                            <div class='lidc-box'>
                            <b>📄 多个LIDC DICOM文件:</b><br>
                            • 选择一个LIDC病例的所有DICOM文件<br>
                            • 系统合并为3D体积进行处理<br>
                            • 确保文件来自同一序列/研究<br>
                            • 保持LIDC原始切片顺序
                            </div>
                            """)

                            dicom_files = gr.File(
                                label="选择LIDC病例的所有DICOM文件",
                                file_types=[".dcm", ".dicom"],
                                file_count="multiple"
                            )

                            process_files_btn = gr.Button("🔍 3D处理文件", variant="secondary", size="lg")

                    gr.HTML("<div class='section-title'>⚙️ 3D检测参数</div>")

                    test_all_versions = gr.Checkbox(
                        label="🧪 启用多版本3D处理",
                        value=True,
                        info="测试多种预处理方法以找到LIDC数据的最优检测"
                    )

                    gr.Markdown("""
                    <div class='warning-box'>
                    <b>💡 3D处理说明:</b><br>
                    • <b>处理时间</b>: 3D分析比2D更耗时（每个病例30秒-2分钟）<br>
                    • <b>内存使用</b>: 完整体积处理需要更多RAM<br>
                    • <b>LIDC适配</b>: 系统自动适配LIDC扫描变化<br>
                    • <b>多版本测试</b>: 为您的特定LIDC数据找到最佳预处理
                    </div>
                    """)

            gr.HTML("<div class='section-title'>🖼️ 3D检测结果</div>")

            with gr.Row():
                detection_result_3d = gr.Plot(
                    label="3D检测结果可视化",
                    show_label=True
                )

                detection_report_3d = gr.Textbox(
                    label="详细3D检测报告",
                    lines=30,
                    max_lines=35,
                    interactive=False,
                    value="""🔄 请加载Bundle并上传LIDC DICOM数据...

📋 3D LIDC检测流程:
1️⃣ 上传MonAI Bundle文件（LUNA16训练模型）
2️⃣ 上传LIDC DICOM数据（ZIP或多个文件）
3️⃣ 启用多版本测试（推荐）
4️⃣ 开始3D处理并等待结果
5️⃣ 查看3D体积分析和检测

🎯 3D vs 2D优势:
• 🌐 完整3D上下文 - 同时分析整个体积
• 🎯 更高准确性 - 利用切片间的空间关系
• 📊 体积分析 - 提供3D测量和体积
• 🔍 减少误报 - 3D形状分析过滤伪影
• 📈 LUNA16兼容 - 与模型训练完全相同的处理

📊 LIDC数据集处理:
• 原始LIDC-IDRI CT扫描（1,018例）
• 可变扫描协议和层厚
• 包含<3mm结节（不在LUNA16训练中）
• 4名放射科医师共识注释
• 通过预处理与LUNA16模型兼容

💡 LIDC的预期结果:
• 高置信度检测≥3mm结节（LUNA16训练重点）
• 可能以较低置信度检测更小结节
• 3D测量（mm³）用于临床评估
• 考虑LIDC扫描协议变化

⚠️ 重要考虑:
• 模型在LUNA16上训练（处理的LIDC子集>3mm）
• LIDC包含比LUNA16训练更多样的数据
• 考虑LIDC解释的置信度阈值
• 如有可能，与放射科医师注释交叉参考"""
                )

            # 事件绑定
            load_bundle_btn.click(
                fn=self.load_bundle_3d,
                inputs=[bundle_file],
                outputs=[bundle_info, bundle_status]
            )

            process_zip_btn.click(
                fn=self.process_dicom_zip_3d,
                inputs=[dicom_zip, test_all_versions],
                outputs=[detection_result_3d, detection_report_3d],
                show_progress=True
            )

            process_files_btn.click(
                fn=self.process_multiple_dicoms_3d,
                inputs=[dicom_files, test_all_versions],
                outputs=[detection_result_3d, detection_report_3d],
                show_progress=True
            )

            gr.Markdown(f"""
            ---
            ### 📋 3D LIDC检测使用指南
            
            #### 🔬 LIDC数据集兼容性:
            
            **什么是LIDC-IDRI？**
            - 包含1,018例的原始肺部CT数据集
            - LUNA16挑战的源数据集
            - 包含所有结节尺寸（包括<3mm）
            - 4名放射科医师共识注释
            
            **LUNA16 vs LIDC关系:**
            ```
            LIDC-IDRI (1,018例) → [过滤>3mm, 预处理] → LUNA16 (888例)
            ```
            
            **您的情况:**
            - ✅ 您有LIDC数据（原始源）
            - ✅ 模型在LUNA16上训练（处理子集）  
            - ✅ 通过预处理流程兼容
            - ✅ 可能检测到其他小结节
            
            #### 🎯 3D检测工作流:
            
            **步骤1: 数据准备**
            ```
            LIDC DICOM → 3D体积重建 → LUNA16预处理
            ↓
            间距: 可变 → 0.703125×0.703125×1.25mm
            尺寸: 可变 → 192×192×80
            方向: 可变 → RAS
            ```
            
            **步骤2: 3D模型推理**
            ```
            3D体积 → RetinaNet 3D → 3D边界框
            ↓
            输出: [x1,y1,z1,x2,y2,z2] + 置信度分数
            ```
            
            **步骤3: LIDC特定后处理**
            ```
            按置信度过滤 → 计算3D体积 → 临床评估
            ```
            
            #### 💡 LIDC的预期结果:
            
            **高置信度检测（>0.5）:**
            - 结节直径≥4mm
            - 清晰的结节形态
            - 符合LUNA16训练标准
            
            **中等置信度检测（0.3-0.5）:**
            - 结节直径3-4mm
            - 边界LUNA16标准
            - 值得临床审查
            
            **低置信度检测（0.1-0.3）:**
            - 结节直径<3mm
            - 不在LUNA16训练中
            - LIDC特定发现
            
            #### 🔧 LIDC优化技巧:
            
            **如果无检测:**
            1. 降低置信度阈值到0.1
            2. 检查DICOM文件完整性
            3. 验证扫描中的肺部解剖
            4. 尝试多版本预处理
            
            **如果误报太多:**
            1. 提高置信度阈值到0.5
            2. 专注于≥4mm的检测
            3. 与放射科医师注释交叉参考
            
            #### 📊 技术规格:
            
            **系统要求:**
            - CPU: 推荐8+核心用于3D处理
            - RAM: 推荐16GB+用于完整体积
            - GPU: 可选但显著加速推理
            
            **支持的LIDC格式:**
            - DICOM文件（.dcm, .dicom）
            - DICOM序列ZIP压缩包
            - 可变层厚（自动处理）
            - 不同扫描协议（标准化）
            
            ---
            
            **版本**: 3D LUNA16/LIDC兼容 v5.0.0  
            **用户**: veryjoyran  
            **时间**: 2025-06-24 15:31:04  
            **数据集**: LIDC-IDRI兼容处理
            
            **关键创新**: LIDC源数据 + LUNA16模型兼容性，通过标准化3D预处理实现
            """)

        return interface


def main():
    """主函数"""
    print("🚀 启动3D LUNA16/LIDC肺结节检测界面")
    print(f"👤 当前用户: veryjoyran")
    print(f"📅 当前时间: 2025-06-24 15:31:04")
    print("🎯 3D LUNA16模型 + LIDC数据兼容性")
    print("=" * 80)

    try:
        app = LUNA16GradioInterface()
        interface = app.create_interface()

        print("✅ 3D LIDC兼容界面创建成功")
        print("📌 主要特性:")
        print("   • 真正的3D体积分析（非逐切片）")
        print("   • LUNA16标准预处理（精确复现训练环境）")
        print("   • LIDC数据兼容（自动处理协议差异）")
        print("   • 多版本预处理测试（找出最佳配置）")
        print("   • 3D空间上下文分析（更高准确性）")
        print("   • 完整中文界面（用户友好）")

        print("\n💡 LIDC数据使用说明:")
        print("   • LIDC是LUNA16的原始数据源，完全兼容")
        print("   • 系统自动处理LIDC与LUNA16的差异")
        print("   • 可能检测到<3mm的小结节（超出LUNA16训练范围）")
        print("   • 建议使用较低置信度阈值查看所有检测")

        print("\n🔧 界面改进:")
        print("   • 修复了AttributeError错误")
        print("   • 添加了中文字体支持")
        print("   • 优化了用户体验和错误处理")
        print("   • 提供了详细的LIDC数据指导")

        interface.launch(
            server_name="127.0.0.1",
            server_port=7866,  # 避免端口冲突
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