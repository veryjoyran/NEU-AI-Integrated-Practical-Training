"""
3D肺结节检测Gradio界面 - 完整增强交互版本
Author: veryjoyran
Date: 2025-06-25 14:03:56
"""

import gradio as gr
import shutil
import tempfile
from pathlib import Path
import zipfile
import os
from datetime import datetime
import matplotlib.pyplot as plt

# 设置matplotlib中文字体支持
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 12
matplotlib.use('Agg')  # 使用非交互式后端

# 导入3D检测器
try:
    from inference_only_3d_detector import Pure3DDetector

    print("✅ 成功导入3D检测器")
except ImportError as e:
    print(f"❌ 导入3D检测器失败: {e}")
    raise


class LUNA16GradioInterface:
    """LUNA16/LIDC兼容的3D检测Gradio界面 - 完整增强版本"""

    def __init__(self):
        self.detector = None
        self.current_bundle_path = None
        self.last_result = None  # 保存最后的检测结果

        print(f"🚀 初始化LUNA16/LIDC Gradio界面 (完整增强版)")
        print(f"   当前用户: veryjoyran")
        print(f"   时间: 2025-06-25 14:03:56")

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
🔧 精确匹配: {model_info.get('exact_matches', 0)}个
⚡ 部分匹配: {model_info.get('partial_matches', 0)}个

🎯 3D LUNA16特性:
  • 3D体积处理（非逐片处理）
  • LUNA16标准预处理流程
  • 体素间距: 0.703125 x 0.703125 x 1.25 mm
  • 模型输入尺寸: 192 x 192 x 80
  • LIDC数据集兼容
  • 多版本预处理测试
  • 🔥 LIDC XML注释回退功能

🔬 LIDC注释回退功能:
  • ✅ 自动查找LIDC XML注释文件
  • ✅ AI无检测时显示人工标注
  • ✅ 解析放射科医师共识注释
  • ✅ 提供完整的注释可视化
  • ✅ 生成对比分析报告

🎨 增强可视化功能:
  • ✅ 大图模式高分辨率显示
  • ✅ 完美中文字体支持
  • ✅ 多视图自动生成
  • ✅ 交互式图片浏览
  • ✅ 颜色编码医师区分

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
        """🔥 处理DICOM ZIP文件 - 确保LIDC注释回退"""
        try:
            if self.detector is None:
                return None, "❌ 请先加载MonAI Bundle"

            if zip_file is None:
                return None, "❌ 请上传DICOM ZIP文件"

            print(f"🔄 处理DICOM ZIP (3D LIDC模式 + XML注释回退): {zip_file.name}")

            # 解压DICOM文件
            temp_dir = Path(tempfile.mkdtemp(prefix="lidc_dicom_"))
            try:
                with zipfile.ZipFile(zip_file.name, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)

                print(f"   ZIP解压到: {temp_dir}")

                # 🔥 查找DICOM文件和XML注释文件
                dicom_files = []
                xml_files = []

                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        file_path = Path(root) / file
                        if file.lower().endswith(('.dcm', '.dicom')):
                            dicom_files.append(file_path)
                        elif file.lower().endswith('.xml'):
                            xml_files.append(file_path)

                print(f"   找到 {len(dicom_files)} 个DICOM文件")
                print(f"   找到 {len(xml_files)} 个XML文件")

                if not dicom_files:
                    return None, "❌ ZIP文件中未找到DICOM文件"

                # 🔥 验证XML文件是否为LIDC格式
                lidc_xml_found = False
                for xml_file in xml_files:
                    if self._is_lidc_xml(xml_file):
                        print(f"   ✅ 发现LIDC XML注释: {xml_file.name}")
                        lidc_xml_found = True
                        break

                if not lidc_xml_found and xml_files:
                    print(f"   ⚠️ 发现XML文件但非LIDC格式")
                elif not xml_files:
                    print(f"   ⚠️ 未发现XML注释文件")

                # 使用DICOM文件目录进行3D检测
                dicom_series_dir = dicom_files[0].parent if len(dicom_files) > 1 else dicom_files[0]

                # 🔥 调用带LIDC回退的检测方法
                return self._process_dicom_3d_with_lidc_fallback(dicom_series_dir, test_all_versions)

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
        """🔥 处理多个DICOM文件 - 确保LIDC注释回退"""
        try:
            if self.detector is None:
                return None, "❌ 请先加载MonAI Bundle"

            if not dicom_files:
                return None, "❌ 请上传DICOM文件"

            print(f"🔄 处理 {len(dicom_files)} 个DICOM文件 (3D LIDC模式 + XML注释回退)")

            # 创建临时目录并复制文件
            temp_dir = Path(tempfile.mkdtemp(prefix="lidc_series_"))
            try:
                for i, file in enumerate(dicom_files):
                    dest_path = temp_dir / f"slice_{i:04d}.dcm"
                    shutil.copy(file.name, dest_path)

                print(f"   文件已复制到: {temp_dir}")

                # 🔥 调用带LIDC回退的检测方法
                return self._process_dicom_3d_with_lidc_fallback(temp_dir, test_all_versions)

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

    def _process_dicom_3d_with_lidc_fallback(self, dicom_path, test_all_versions):
        """🔥 3D DICOM处理核心函数 - 强制使用LIDC注释回退"""
        try:
            print(f"🔍 开始3D LIDC检测 (强制LIDC注释回退)...")
            print(f"   输入: {dicom_path}")
            print(f"   测试所有版本: {test_all_versions}")

            # 🔥 强制使用带LIDC回退的检测方法
            result = self.detector.detect_3d_with_lidc_fallback(
                dicom_path,
                test_all_versions=test_all_versions
            )

            if result is None:
                return None, """❌ 3D检测和LIDC注释回退均失败

💡 可能的问题:
• DICOM文件损坏或格式不支持
• 系统无法找到对应的LIDC XML注释文件
• XML文件格式不符合LIDC标准
• DICOM数据无法转换为3D体积

🔧 故障排除步骤:
• 确保ZIP文件包含完整的DICOM序列
• 验证ZIP文件中包含LIDC XML注释文件
• 检查XML文件是否包含'LidcReadMessage'或'readingSession'标签
• 尝试重新下载LIDC数据集
• 联系技术支持获取帮助"""

            # 保存结果用于后续交互
            self.last_result = result

            # 🔥 检查是否使用了LIDC注释回退
            using_lidc_fallback = result.get('lidc_fallback_used', False)
            ai_detection_count = result.get('detection_count', 0)

            if using_lidc_fallback:
                print(f"✅ LIDC注释回退已激活，显示 {ai_detection_count} 个注释结节")
                status_msg = f"🔄 AI检测无结果，已启用LIDC注释回退显示 {ai_detection_count} 个人工标注结节"
            else:
                if ai_detection_count > 0:
                    print(f"✅ AI检测成功，发现 {ai_detection_count} 个候选结节")
                    status_msg = f"✅ AI检测成功，发现 {ai_detection_count} 个候选结节"
                else:
                    print("❌ AI检测无结果，且未找到LIDC注释")
                    status_msg = "❌ AI检测无结果，且未找到可用的LIDC XML注释文件"

            # 生成可视化
            print("🎨 生成增强检测结果可视化...")
            fig = self.detector.visualize_3d_result(result)

            # 生成详细报告
            print("📝 生成详细检测报告...")
            report = self.detector.generate_3d_report(result, dicom_path)

            # 🔥 在报告前添加状态说明和交互提示
            final_report = f"""
🔔 检测状态: {status_msg}

💡 可视化交互提示:
• 图片可能在浏览器中显示较小，建议右键选择"在新标签页中打开图片"查看原始大图
• 系统已生成高分辨率图像（200 DPI），支持放大查看细节
• 图片中的中文标注已优化字体显示
• 所有视图已自动保存，可右键"图片另存为"保存到本地

🎨 可视化说明:
• 🔴 红色区域：第1位放射科医师的标注
• 🔵 蓝色区域：第2位放射科医师的标注  
• 🟢 绿色区域：第3位放射科医师的标注
• 🟠 橙色区域：第4位放射科医师的标注
• 虚线框：结节边界框
• 实线轮廓：精确结节边界
• 标签显示：结节ID + 恶性程度(M) + 细微程度(S) + 医师ID

{report}
"""

            return fig, final_report

        except Exception as e:
            error_msg = f"❌ 3D DICOM处理失败: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return None, error_msg

    def _is_lidc_xml(self, xml_path):
        """🔥 验证是否为LIDC XML文件"""
        try:
            with open(xml_path, 'r', encoding='utf-8') as f:
                content = f.read(1000)  # 读取前1000字符
                is_lidc = ('LidcReadMessage' in content or
                           'readingSession' in content or
                           'unblindedReadNodule' in content or
                           'blindedReadNodule' in content)

                if is_lidc:
                    print(f"   ✅ 验证LIDC XML格式: {xml_path.name}")
                else:
                    print(f"   ❌ 非LIDC XML格式: {xml_path.name}")

                return is_lidc
        except Exception as e:
            print(f"   ⚠️ XML文件验证失败: {e}")
            return False

    def create_interface(self):
        """🔥 创建完整增强的3D Gradio界面"""

        custom_css = """
        .main-title { 
            font-size: 28px; 
            font-weight: bold; 
            text-align: center; 
            margin-bottom: 25px; 
            color: #2c3e50; 
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        .section-title { 
            font-size: 20px; 
            font-weight: bold; 
            margin-top: 20px; 
            margin-bottom: 15px; 
            color: #34495e; 
            border-bottom: 2px solid #3498db;
            padding-bottom: 5px;
        }
        .info-box { 
            background: linear-gradient(135deg, #e8f6f3 0%, #d5f4e6 100%);
            padding: 20px; 
            border-radius: 12px; 
            margin: 15px 0; 
            border-left: 5px solid #1abc9c; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .warning-box { 
            background: linear-gradient(135deg, #fdf2e9 0%, #fdeaa7 100%);
            padding: 20px; 
            border-radius: 12px; 
            margin: 15px 0; 
            border-left: 5px solid #e67e22; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .lidc-box { 
            background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%);
            padding: 20px; 
            border-radius: 12px; 
            margin: 15px 0; 
            border-left: 5px solid #4169e1; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .xml-box { 
            background: linear-gradient(135deg, #f5f5dc 0%, #f0e68c 100%);
            padding: 20px; 
            border-radius: 12px; 
            margin: 15px 0; 
            border-left: 5px solid #ff6347; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .viz-box { 
            background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
            padding: 20px; 
            border-radius: 12px; 
            margin: 15px 0; 
            border-left: 5px solid #4caf50; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .large-plot {
            min-height: 700px !important;
            width: 100% !important;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
        .enhanced-textarea {
            font-family: 'Courier New', monospace;
            line-height: 1.6;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .3d-badge { 
            background: linear-gradient(45deg, #e74c3c, #c0392b);
            color: white; 
            padding: 6px 12px; 
            border-radius: 20px; 
            font-size: 14px; 
            font-weight: bold;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        .lidc-badge { 
            background: linear-gradient(45deg, #4169e1, #1e3a8a);
            color: white; 
            padding: 6px 12px; 
            border-radius: 20px; 
            font-size: 14px; 
            font-weight: bold;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        .xml-badge { 
            background: linear-gradient(45deg, #ff6347, #dc2626);
            color: white; 
            padding: 6px 12px; 
            border-radius: 20px; 
            font-size: 14px; 
            font-weight: bold;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        .enhanced-badge { 
            background: linear-gradient(45deg, #27ae60, #16a085);
            color: white; 
            padding: 6px 12px; 
            border-radius: 20px; 
            font-size: 14px; 
            font-weight: bold;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        .interact-badge { 
            background: linear-gradient(45deg, #8e44ad, #6c1a97);
            color: white; 
            padding: 6px 12px; 
            border-radius: 20px; 
            font-size: 14px; 
            font-weight: bold;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        .complete-badge { 
            background: linear-gradient(45deg, #f39c12, #d68910);
            color: white; 
            padding: 6px 12px; 
            border-radius: 20px; 
            font-size: 14px; 
            font-weight: bold;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        """

        with gr.Blocks(title="3D LUNA16/LIDC肺结节检测 - 完整增强版", css=custom_css,
                       theme=gr.themes.Soft()) as interface:
            gr.HTML("""
            <div class='main-title'>
                🫁 3D LUNA16/LIDC肺结节检测系统 
                <br><br>
                <span class='3d-badge'>3D v5.3.0</span>
                <span class='lidc-badge'>LIDC兼容</span>
                <span class='xml-badge'>XML回退</span>
                <span class='enhanced-badge'>增强可视化</span>
                <span class='interact-badge'>交互增强</span>
                <span class='complete-badge'>完整版</span>
            </div>
            """)

            gr.Markdown("""
            <div class='info-box'>
            <b>🔥 完整增强版本特性 (v5.3.0):</b><br>
            • ✅ <b>真正的3D处理</b>: 完整的体积分析，而非逐片处理<br>
            • ✅ <b>LUNA16标准</b>: 与模型训练完全一致的预处理（0.703125mm间距，192×192×80）<br>
            • ✅ <b>LIDC兼容</b>: 处理原始LIDC-IDRI数据集变化<br>
            • ✅ <b>多版本测试</b>: 自动测试最优预处理方法<br>
            • ✅ <b>3D上下文</b>: 利用所有切片间的空间关系<br>
            • 🔥 <b>LIDC XML注释回退</b>: AI无检测时自动显示人工标注真值<br>
            • 🎨 <b>大图高清显示</b>: 200 DPI高分辨率，解决图片过小问题<br>
            • 🈷️ <b>完美中文支持</b>: 修正字体显示，完美显示中文标注<br>
            • 🌈 <b>颜色编码可视化</b>: 不同颜色区分4位放射科医师注释<br>
            • 📊 <b>多视图生成</b>: 概览、详细、分析、对比四个专业视图
            </div>
            """)

            gr.Markdown("""
            <div class='xml-box'>
            <b>🔥 LIDC XML注释回退功能 (完整版):</b><br>
            • <b>智能激活</b>: 当AI检测无结果时，系统自动查找并解析LIDC XML注释文件<br>
            • <b>专业解析</b>: 完整解析4名放射科医师的详细注释，包括结节特征、位置、恶性程度<br>
            • <b>高清可视化</b>: 将人工标注转换为高分辨率可视化，支持轮廓、边界框、标签三重显示<br>
            • <b>颜色编码</b>: 红色(医师1)、蓝色(医师2)、绿色(医师3)、橙色(医师4)区分不同医师<br>
            • <b>交互增强</b>: 右键图片可在新标签页查看原始大图，支持放大查看细节<br>
            • <b>多视图分析</b>: 自动生成概览、详细注释、统计分析、医师对比四个专业视图<br>
            • <b>对比报告</b>: 生成AI检测与人工注释的详细对比分析报告
            </div>
            """)

            gr.Markdown("""
            <div class='viz-box'>
            <b>🎨 可视化增强说明 (完整版):</b><br>
            • <b>大图模式</b>: 图片尺寸优化为16×12或18×10，提供清晰的细节显示<br>
            • <b>高分辨率</b>: 200 DPI渲染，确保图片放大后依然清晰<br>
            • <b>中文字体修正</b>: 支持SimHei、Microsoft YaHei，完美显示中文标注<br>
            • <b>交互功能</b>: 右键菜单支持"在新标签页中打开图片"、"图片另存为"<br>
            • <b>专业标注</b>: 每个结节显示ID、恶性程度(M)、细微程度(S)、医师ID<br>
            • <b>多层显示</b>: 轮廓线条 + 边界框 + 文字标签三重显示确保信息完整
            </div>
            """)

            gr.Markdown("""
            <div class='lidc-box'>
            <b>📊 LIDC数据集兼容性说明 (完整版):</b><br>
            • <b>数据源关系</b>: LIDC-IDRI是LUNA16派生的原始数据集，包含1,018例完整CT扫描<br>
            • <b>数据差异</b>: LIDC包含所有结节（包括<3mm），LUNA16只过滤到>3mm结节<br>
            • <b>模型兼容性</b>: 模型在LUNA16子集上训练，通过标准化预处理与LIDC源数据完全兼容<br>
            • <b>检测能力</b>: 可能检测到比LUNA16挑战更小的结节，系统会调整置信度阈值显示<br>
            • <b>注释标准</b>: LIDC有4名放射科医师共识注释，LUNA16有统一处理的真值标准<br>
            • <b>XML回退优势</b>: 当AI无检测时，显示LIDC人工标注作为医学专业参考标准<br>
            • <b>临床价值</b>: 可用于评估AI模型性能，提供医师标注的临床决策参考
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
                    <b>💡 3D Bundle要求 (完整版):</b><br>
                    • lung_nodule_ct_detection_v0.5.x.zip (LUNA16训练)<br>
                    • RetinaNet或兼容的3D检测模型<br>
                    • 自动3D模型加载和验证<br>
                    • LIDC兼容预处理流程<br>
                    • 权重加载成功率修复<br>
                    • 支持精确匹配和部分匹配权重加载
                    </div>
                    """)

                    load_bundle_btn = gr.Button("🚀 加载3D Bundle", variant="primary", size="lg")

                    bundle_status = gr.Textbox(
                        label="Bundle状态",
                        value="未加载",
                        interactive=False,
                        lines=1
                    )

                    bundle_info = gr.Textbox(
                        label="3D Bundle详细信息",
                        lines=25,
                        interactive=False,
                        elem_classes=["enhanced-textarea"],
                        value="🔄 请上传MonAI Bundle文件开始3D LIDC检测..."
                    )

                with gr.Column(scale=1):
                    gr.HTML("<div class='section-title'>📁 LIDC DICOM数据上传</div>")

                    with gr.Tabs():
                        with gr.TabItem("🗂️ ZIP压缩包（包含XML注释）"):
                            gr.Markdown("""
                            <div class='xml-box'>
                            <b>📦 LIDC DICOM + XML ZIP上传 (完整版):</b><br>
                            • 上传包含DICOM文件和XML注释的完整ZIP包<br>
                            • 系统自动查找和解析LIDC XML注释文件<br>
                            • AI无检测时自动显示4名放射科医师的人工标注真值<br>
                            • 提供AI检测与人工注释的详细对比分析<br>
                            • 🔥 <b>重要</b>: 确保ZIP中包含LIDC标准XML文件<br>
                            • 🎨 <b>增强</b>: 高清可视化显示，完美中文字体支持<br>
                            • 📊 <b>多视图</b>: 自动生成概览、详细、分析、对比四个视图
                            </div>
                            """)

                            dicom_zip = gr.File(
                                label="上传LIDC DICOM序列ZIP (含XML注释)",
                                file_types=[".zip"],
                                file_count="single"
                            )

                            process_zip_btn = gr.Button("🔍 3D处理ZIP (含XML回退)", variant="primary", size="lg")

                        with gr.TabItem("📄 多个DICOM文件"):
                            gr.Markdown("""
                            <div class='lidc-box'>
                            <b>📄 多个LIDC DICOM文件 (完整版):</b><br>
                            • 选择一个LIDC病例的所有DICOM文件<br>
                            • 系统合并为3D体积进行处理<br>
                            • 如有XML注释文件，请一并上传<br>
                            • 保持LIDC原始切片顺序<br>
                            • 支持可变层厚和扫描协议<br>
                            • 自动适配LUNA16标准预处理
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
                        info="测试多种预处理方法以找到LIDC数据的最优检测配置"
                    )

                    gr.Markdown("""
                    <div class='warning-box'>
                    <b>💡 3D处理说明 (完整版):</b><br>
                    • <b>处理时间</b>: 3D分析比2D更耗时（每个病例30秒-3分钟）<br>
                    • <b>内存使用</b>: 完整体积处理需要更多RAM（建议16GB+）<br>
                    • <b>LIDC适配</b>: 系统自动适配LIDC扫描变化和协议差异<br>
                    • <b>XML回退</b>: AI无检测时自动显示人工标注真值<br>
                    • <b>多版本测试</b>: 为您的特定LIDC数据找到最佳预处理方法<br>
                    • <b>可视化增强</b>: 高分辨率图像生成，支持交互式查看
                    </div>
                    """)

            gr.HTML("<div class='section-title'>🖼️ 完整增强3D检测结果可视化</div>")

            gr.Markdown("""
            <div class='viz-box'>
            <b>🎨 可视化功能说明 (完整版):</b><br>
            • <b>大图显示</b>: 图片尺寸优化，提供清晰的细节显示，解决图片过小问题<br>
            • <b>高分辨率</b>: 200 DPI渲染，图片放大后依然清晰，适合医学影像分析<br>
            • <b>中文字体</b>: 完美支持中文标注显示，使用SimHei、Microsoft YaHei字体<br>
            • <b>交互功能</b>: 右键图片选择"在新标签页中打开"可查看原始大图<br>
            • <b>颜色编码</b>: 🔴红色 🔵蓝色 🟢绿色 🟠橙色 区分4位放射科医师<br>
            • <b>多层标注</b>: 轮廓线 + 边界框 + 文字标签 三重显示确保信息完整<br>
            • <b>专业信息</b>: 每个结节显示ID、恶性程度(M:1-5)、细微程度(S:1-5)、医师ID
            </div>
            """)

            with gr.Row():
                # 🔥 主可视化区域 - 更大的显示空间
                detection_result_3d = gr.Plot(
                    label="🎨 3D检测结果主视图 (完整增强可视化)",
                    show_label=True,
                    elem_classes=["large-plot"]
                )

            with gr.Row():
                # 详细报告区域
                detection_report_3d = gr.Textbox(
                    label="📝 详细3D检测报告 (完整版含交互提示)",
                    lines=30,
                    max_lines=35,
                    interactive=False,
                    elem_classes=["enhanced-textarea"],
                    value="""🔄 请加载Bundle并上传LIDC DICOM数据...

📋 完整3D LIDC检测流程 (v5.3.0):
1️⃣ 上传MonAI Bundle文件（LUNA16训练模型）
2️⃣ 上传包含XML注释的LIDC DICOM ZIP文件
3️⃣ 启用多版本测试（推荐，找出最佳配置）
4️⃣ 开始3D处理并等待结果
5️⃣ 查看AI检测结果或LIDC注释回退可视化

🔥 完整版LIDC XML注释回退功能:
• 🤖 AI检测优先: 首先尝试AI模型检测结节
• 📋 智能回退: AI无检测时自动查找并解析XML注释
• 🔍 专业解析: 完整解析4名放射科医师的详细标注
• 📊 多维对比: 生成AI vs 人工注释的详细对比报告
• 🎨 高清可视化: 200 DPI高分辨率，解决图片过小问题
• 🈷️ 中文支持: 完美显示中文标注，字体显示问题已修正

🎯 完整版LIDC XML注释内容解析:
• 📍 精确位置: 结节在CT扫描中的3D坐标和轮廓
• 🏥 医师信息: 4名放射科医师的独立标注和共识
• 📊 恶性程度: 1-5级评分（1=高度良性，5=高度恶性）
• 🔍 细微程度: 1-5级评分（1=极其细微，5=极其明显）
• 📏 详细特征: 内部结构、钙化、球形度、边缘、纹理等9项特征
• 📐 精确测量: 结节面积、体积、3D尺寸等量化指标

🎨 完整版可视化增强功能:
• 🖼️ 大图模式: 图片尺寸16×12，提供清晰细节显示
• 🎯 智能切片: 自动选择有注释的切片进行显示
• 🌈 颜色编码: 红(医师1) 蓝(医师2) 绿(医师3) 橙(医师4)
• 📝 详细标签: 结节ID + M:恶性程度 + S:细微程度 + 医师ID
• 🔲 双重显示: 实线轮廓 + 虚线边界框确保位置准确
• 📊 多视图生成: 概览/详细/分析/对比四个专业视图

🔧 完整版交互功能:
• 🔍 图片放大: 右键"在新标签页中打开图片"查看原始大图
• 💾 图片保存: 右键"图片另存为"保存200 DPI高清图片
• 📱 响应式显示: 适配不同屏幕尺寸和分辨率
• 🎨 字体优化: 自动选择最佳中文字体确保显示效果
• 📊 实时反馈: 详细的处理过程和结果状态显示

💡 完整版使用技巧:
• 📦 文件准备: 确保ZIP包含完整DICOM序列和LIDC XML注释
• ⚙️ 参数设置: 启用多版本处理获得最佳检测效果  
• 🖼️ 查看技巧: 图片可能在浏览器中显示较小，右键新标签页打开查看大图
• 📝 报告分析: 查看详细报告获取量化分析和临床建议
• 🏥 医师对比: 关注不同医师标注的一致性和差异性
• 📊 风险评估: 重点关注恶性程度≥4的结节进行进一步分析

⚠️ 完整版重要说明:
• 🎯 检测优先级: 系统优先显示AI检测结果，仅在无检测时启用XML回退
• 🏥 医学标准: LIDC注释代表4名放射科医师的专业判断和共识
• 📊 评估工具: 可用于评估AI模型性能和临床决策参考
• 🔬 研究用途: 适用于医学影像研究和AI模型开发验证
• ⚖️ 责任声明: 本系统仅供研究和教育用途，不可替代临床诊断

📞 技术支持: veryjoyran | 完整增强版 v5.3.0
📅 更新时间: 2025-06-25 14:03:56
🏷️ 特色功能: 大图可视化 + 中文字体修正 + XML注释回退 + 多视图生成 + 交互增强"""
                )

            # 🔥 添加可视化控制和说明区域
            with gr.Row():
                gr.Markdown("""
                <div class='viz-box'>
                <b>🎛️ 完整版可视化交互指南:</b><br>
                • <b>🔍 图片放大查看</b>: 右键点击图片 → "在新标签页中打开图片" → 查看200 DPI原始高分辨率版本<br>
                • <b>📊 多视图浏览</b>: 系统自动生成概览、详细注释、统计分析、医师对比四个专业视图<br>
                • <b>💾 图片保存功能</b>: 右键图片 → "图片另存为" → 保存高清版本到本地进行进一步分析<br>
                • <b>🈷️ 中文字体显示</b>: 系统已优化中文字体支持，如显示异常请确保浏览器支持中文字体<br>
                • <b>🌈 颜色编码说明</b>: 🔴红色=医师1 🔵蓝色=医师2 🟢绿色=医师3 🟠橙色=医师4<br>
                • <b>📏 标注信息解读</b>: 每个标签显示"结节ID + M:恶性程度(1-5) + S:细微程度(1-5) + 医师ID"<br>
                • <b>📱 响应式设计</b>: 界面自适应不同屏幕尺寸，在大屏幕上获得最佳体验
                </div>
                """)

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
            ### 📋 完整版 LIDC XML注释回退功能详细说明

            #### 🔥 核心创新功能:

            **智能回退机制:**
            ```
            用户上传数据 → AI 3D检测 → 检测结果判断
                                           ↓
            检测到结节 ← 显示AI结果    无检测结果
                                           ↓
                              查找LIDC XML注释文件
                                           ↓
                              解析4名医师标注
                                           ↓
                              生成高清可视化
                                           ↓
            显示XML注释回退 ← 解析成功    解析失败 → 显示详细错误信息
            ```

            **支持的完整XML格式:**
            - ✅ LIDC标准XML格式 (LidcReadMessage)
            - ✅ 包含readingSession标签
            - ✅ 包含unblindedReadNodule和blindedReadNodule标签
            - ✅ 完整的结节特征数据 (恶性程度、细微程度等)
            - ✅ 精确的ROI轮廓坐标数据
            - ✅ 4名放射科医师的独立标注

            #### 📊 完整回退显示内容:

            **基本统计信息:**
            - 参与放射科医师数量 (通常为4名)
            - 每位医师的标注结节总数
            - XML注释文件版本信息
            - 标注时间和系统信息

            **详细结节信息:**
            - 结节唯一标识ID和名称
            - 3D位置坐标 (Z轴位置)
            - 精确的结节边界轮廓坐标点
            - 结节边界框 (x_min, y_min, x_max, y_max)
            - 计算得出的面积和估算体积

            **专业医学特征:**
            - 恶性程度评分 (1-5): 1=高度良性, 5=高度恶性
            - 细微程度评分 (1-5): 1=极其细微, 5=极其明显
            - 内部结构评分 (1-6): 软组织、钙化、脂肪等
            - 球形度评分 (1-5): 结节的球形程度
            - 边缘评分 (1-5): 边缘的清晰度和规则性
            - 纹理评分 (1-5): 结节的纹理特征

            **高清可视化元素:**
            - 🎨 彩色轮廓线条 (区分不同医师)
            - 📐 虚线边界框 (精确位置标示)
            - 🏷️ 详细信息标签 (ID + 特征评分)
            - 🌈 颜色编码系统 (4种颜色对应4名医师)
            - 📊 透明度填充 (区域范围可视化)

            #### 💡 完整版使用最佳实践:

            **文件准备建议:**
            ```
            推荐的LIDC ZIP文件结构:
            LIDC_Case_XXXX.zip
            ├── DICOM序列文件夹/
            │   ├── slice_001.dcm
            │   ├── slice_002.dcm
            │   ├── ...
            │   └── slice_XXX.dcm
            └── LIDC注释文件.xml (包含LidcReadMessage)
            ```

            **检测流程优化:**
            1. 📤 上传包含XML的完整LIDC ZIP文件
            2. ⚙️ 启用多版本处理获得最佳检测效果
            3. ⏳ 等待AI检测完成 (30秒-3分钟)
            4. 🔍 如AI无检测，系统自动启用XML注释回退
            5. 🎨 查看高分辨率可视化结果
            6. 📝 阅读详细对比分析报告

            **可视化查看技巧:**
            - 🖼️ 主视图显示最重要的注释切片
            - 🔍 右键"在新标签页打开"查看原始大图
            - 💾 右键"图片另存为"保存高清版本
            - 📊 系统自动生成多个分析视图
            - 🌈 颜色编码帮助区分不同医师标注

            #### 📈 完整版临床应用价值:

            **AI性能评估:**
            - 对比AI检测与4名医师共识的差异
            - 分析AI模型的敏感性和特异性
            - 识别AI可能遗漏的结节类型
            - 评估不同预处理方法的效果

            **医学教育和研究:**
            - 展示标准化的结节标注流程
            - 比较不同医师的标注风格和一致性
            - 提供真实的临床标注案例学习
            - 支持医学影像AI算法研究开发

            **临床决策支持:**
            - 提供专业医师标注作为参考标准
            - 展示结节的详细特征评分
            - 支持多医师共识的诊断方法
            - 量化结节的恶性风险等级

            ---

            **版本**: 3D LUNA16/LIDC兼容 v5.3.0 (完整增强版)  
            **开发者**: veryjoyran  
            **更新时间**: 2025-06-25 14:03:56  
            **核心特色**: AI检测 + LIDC XML注释回退 + 大图可视化 + 中文字体修正 + 交互增强

            **重大突破**: 完美解决图片过小和中文字体显示问题，实现专业级医学影像可视化，
            提供完整的LIDC注释回退机制，支持4名放射科医师标注的高清显示和交互分析。
            """)

        return interface


def main():
    """主函数 - 完整增强版本"""
    print("🚀 启动3D LUNA16/LIDC肺结节检测界面 (完整增强版)")
    print(f"👤 当前用户: veryjoyran")
    print(f"📅 当前时间: 2025-06-25 14:03:56")
    print("🎯 完整功能 + 大图可视化 + 中文字体修正 + XML注释回退 + 交互增强")
    print("=" * 90)

    try:
        app = LUNA16GradioInterface()
        interface = app.create_interface()

        print("✅ 完整增强界面创建成功")
        print("📌 完整版特性:")
        print("   • 🖼️ 大图模式 - 16×12高分辨率显示，解决图片过小问题")
        print("   • 🈷️ 中文字体完美支持 - SimHei/Microsoft YaHei字体，显示问题已修正")
        print("   • 🎨 200 DPI高清渲染 - 图片放大后依然清晰")
        print("   • 📊 多视图自动生成 - 概览/详细/分析/对比四个专业视图")
        print("   • 🔍 交互功能增强 - 右键放大、保存功能完整支持")
        print("   • 🌈 颜色编码可视化 - 红蓝绿橙区分4名医师注释")
        print("   • 🔥 LIDC XML注释回退 - AI无检测时自动显示人工标注")
        print("   • 📝 详细对比报告 - AI vs 医师标注性能分析")
        print("   • 🏥 专业医学标准 - 完整的LIDC特征评分显示")
        print("   • 🛠️ 鲁棒错误处理 - 完善的异常处理和用户提示")

        print("\n🔥 完整版创新功能:")
        print("   • 智能XML解析 - 自动识别和解析LIDC标准格式")
        print("   • 多医师标注对比 - 4名放射科医师的独立标注显示")
        print("   • 高精度坐标映射 - 精确的结节位置和轮廓显示")
        print("   • 专业特征评分 - 恶性程度、细微程度等9项医学特征")
        print("   • 自适应切片选择 - 智能选择有注释的最佳切片显示")
        print("   • 响应式界面设计 - 适配不同屏幕尺寸和分辨率")

        print("\n🎨 可视化增强亮点:")
        print("   • 图片尺寸问题彻底解决 - 从小图变为清晰大图")
        print("   • 中文字体显示完美 - 告别乱码和字体缺失")
        print("   • 医学标注专业化 - 符合临床阅片习惯")
        print("   • 交互体验现代化 - 支持现代浏览器交互功能")

        interface.launch(
            server_name="127.0.0.1",
            server_port=7869,  # 新端口避免冲突
            debug=True,
            show_error=True,
            inbrowser=True,
            share=False,
            favicon_path=None,
            auth=None
        )

    except Exception as e:
        print(f"❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()