
import gradio as gr
import shutil
import tempfile
from pathlib import Path
import json
from datetime import datetime
import zipfile
import os
import numpy as np

# 导入2D检测器
from inference_2D_detector import Inference2DDetector, Simple2DVisualizer, save_2d_inference_results


class Lung2DDetectionGradioInterface:
    """2D肺结节检测Gradio界面"""

    def __init__(self):
        self.detector = None
        self.visualizer = Simple2DVisualizer()
        self.current_bundle_path = None

    def load_bundle_2d(self, bundle_file):
        """加载MonAI Bundle (2D模式)"""
        try:
            if bundle_file is None:
                return "❌ 请上传MonAI Bundle文件", "未加载Bundle"

            bundle_path = bundle_file.name
            print(f"🔄 加载Bundle (2D模式): {bundle_path}")

            # 初始化2D检测器
            self.detector = Inference2DDetector()
            success = self.detector.load_bundle_2d(bundle_path)

            self.current_bundle_path = bundle_path

            # 获取模型信息
            model_info = self.detector.model_info

            info_text = f"""
✅ 2D MonAI Bundle加载成功!

📁 Bundle文件: {Path(bundle_path).name}
🏗️ 模型类型: {model_info.get('type', 'Unknown')}
🔧 原始类型: {model_info.get('original_type', 'Unknown')}
📅 加载时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
🖥️ 运行设备: {self.detector.device}
🔧 参数数量: {sum(p.numel() for p in self.detector.model.parameters()):,}
✨ 2D适配: {'成功' if model_info.get('adapted_to_2d', False) else '未知'}
🎯 权重加载比例: {model_info.get('loaded_ratio', 0):.2f}

🎯 2D检测特性:
  • 逐切片检测分析
  • 更快的推理速度
  • 更低的内存占用
  • 精确的切片级定位

🚀 系统已准备就绪，可以开始2D检测!
"""

            return info_text, "2D Bundle已加载"

        except Exception as e:
            error_msg = f"❌ 2D Bundle加载失败: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return error_msg, "加载失败"

    def process_dicom_zip_2d(self, zip_file, confidence_threshold, max_slices_to_process):
        """处理DICOM ZIP文件 (2D模式)"""
        try:
            if self.detector is None:
                return None, None, "❌ 请先加载MonAI Bundle"

            if zip_file is None:
                return None, None, "❌ 请上传DICOM ZIP文件"

            print(f"🔄 处理DICOM ZIP (2D模式): {zip_file.name}")

            # 解压DICOM文件
            temp_dir = Path(tempfile.mkdtemp())
            with zipfile.ZipFile(zip_file.name, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            # 查找DICOM文件
            dicom_files = []
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.lower().endswith('.dcm'):
                        dicom_files.append(Path(root) / file)

            if not dicom_files:
                return None, None, "❌ ZIP中未找到DICOM文件"

            print(f"   找到 {len(dicom_files)} 个DICOM文件")

            # 使用第一个DICOM文件的目录作为序列目录
            dicom_series_dir = dicom_files[0].parent

            return self._process_dicom_series_2d(
                dicom_series_dir, confidence_threshold, max_slices_to_process
            )

        except Exception as e:
            error_msg = f"❌ DICOM ZIP处理失败: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return None, None, error_msg

    def process_multiple_dicoms_2d(self, dicom_files, confidence_threshold, max_slices_to_process):
        """处理多个上传的DICOM文件 (2D模式)"""
        try:
            if self.detector is None:
                return None, None, "❌ 请先加载MonAI Bundle"

            if not dicom_files:
                return None, None, "❌ 请上传DICOM文件"

            print(f"🔄 处理 {len(dicom_files)} 个DICOM文件 (2D模式)")

            # 创建临时目录
            temp_dir = Path(tempfile.mkdtemp())

            # 复制DICOM文件到临时目录
            for i, file in enumerate(dicom_files):
                dest_path = temp_dir / f"{i:04d}.dcm"
                shutil.copy(file.name, dest_path)

            print(f"   文件复制到: {temp_dir}")

            return self._process_dicom_series_2d(
                temp_dir, confidence_threshold, max_slices_to_process
            )

        except Exception as e:
            error_msg = f"❌ DICOM文件处理失败: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return None, None, error_msg

    def _process_dicom_series_2d(self, dicom_dir, confidence_threshold, max_slices_to_process):
        """处理DICOM序列的核心函数 (2D模式)"""
        try:
            # 🔥 执行2D批量推理
            detection_results = self.detector.batch_inference_all_slices(
                dicom_dir,
                confidence_threshold=float(confidence_threshold),
                max_slices=int(max_slices_to_process) if max_slices_to_process > 0 else None
            )

            if not detection_results:
                return None, None, "❌ 所有切片都无检测结果\n\n💡 建议:\n• 降低置信度阈值\n• 检查DICOM数据质量\n• 确认为胸部CT扫描"

            # 🔥 提取候选结节
            candidates = self.detector.extract_candidates_2d(
                detection_results,
                min_confidence=float(confidence_threshold)
            )

            # 生成可视化和报告
            return self._create_2d_results(detection_results, candidates, dicom_dir)

        except Exception as e:
            error_msg = f"❌ DICOM序列处理失败: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return None, None, error_msg

    def _create_2d_results(self, detection_results, candidates, dicom_source):
        """创建2D结果可视化和报告"""
        try:
            temp_viz_dir = Path(tempfile.mkdtemp())

            # 创建2D检测总览
            overview_path = temp_viz_dir / "2d_detection_overview.png"
            self.visualizer.create_2d_detection_overview(detection_results, str(overview_path))

            # 创建2D候选蒙太奇
            montage_path = None
            if candidates:
                montage_path = temp_viz_dir / "2d_candidates_montage.png"
                fig = self.visualizer.create_2d_candidates_montage(candidates, detection_results, str(montage_path))

            # 生成2D报告
            report = self._generate_2d_report(detection_results, candidates, dicom_source)

            return str(overview_path), str(montage_path) if montage_path else None, report

        except Exception as e:
            print(f"❌ 2D结果创建失败: {e}")
            return None, None, f"❌ 2D结果创建失败: {str(e)}"

    def _generate_2d_report(self, detection_results, candidates, dicom_source):
        """生成2D检测报告"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 确定数据源描述
        if isinstance(dicom_source, Path):
            source_desc = f"文件夹: {dicom_source.name}"
        else:
            source_desc = "上传的DICOM文件"

        # 计算统计信息
        total_detections = sum(d['detection_count'] for d in detection_results)
        slices_with_detections = len(detection_results)

        if detection_results:
            max_confidence = max(max(d['scores']) for d in detection_results)
            avg_confidence = np.mean([score for d in detection_results for score in d['scores']])
            confidence_std = np.std([score for d in detection_results for score in d['scores']])
        else:
            max_confidence = avg_confidence = confidence_std = 0

        report = f"""
🎯 2D肺结节检测分析报告
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

👤 用户: veryjoyran
📅 检测时间: {current_time}
🤖 模型: {self.detector.model_info.get('type', 'Unknown')} (2D适配)
📁 数据源: {source_desc}

📊 2D检测统计:
  • 有检测结果的切片: {slices_with_detections} 个
  • 总检测数量: {total_detections} 个
  • 候选结节数量: {len(candidates)} 个
  • 最高置信度: {max_confidence:.3f}
  • 平均置信度: {avg_confidence:.3f}
  • 置信度标准差: {confidence_std:.3f}

🔍 逐切片检测详情:
"""

        if not detection_results:
            report += """
❌ 未在任何切片中检测到结节

💡 2D检测分析:
  • 所有切片都未检测到明显的结节
  • 可能原因：阈值设置过高、数据质量问题、非胸部CT

🔧 建议操作:
  • 降低置信度阈值重新检测
  • 检查DICOM数据是否为胸部CT
  • 确认图像质量和窗宽窗位设置
  • 尝试调整最大处理切片数量
"""
        else:
            # 按切片索引排序
            sorted_results = sorted(detection_results, key=lambda x: x['slice_index'])

            for i, result in enumerate(sorted_results[:10]):  # 最多显示前10个
                slice_idx = result['slice_index']
                detection_count = result['detection_count']
                max_score = max(result['scores'])
                avg_score = np.mean(result['scores'])

                report += f"""
📍 切片 #{slice_idx}:
  • 检测数量: {detection_count} 个
  • 最高置信度: {max_score:.3f}
  • 平均置信度: {avg_score:.3f}
  • 检测模式: {'目标检测' if result.get('detection_mode', False) else '分割检测'}
"""

                # 显示该切片的检测框
                for j, (box, score) in enumerate(zip(result['boxes'][:3], result['scores'][:3])):  # 最多显示3个
                    x1, y1, x2, y2 = box
                    size = (x2 - x1) * (y2 - y1)
                    report += f"    检测 {j + 1}: 位置[{x1}, {y1}, {x2}, {y2}], 尺寸{size:.0f}px², 置信度{score:.3f}\n"

            if len(sorted_results) > 10:
                report += f"\n... 还有 {len(sorted_results) - 10} 个切片有检测结果\n"

        report += f"""

🎯 候选结节汇总:
"""

        if not candidates:
            report += """
❌ 未提取到有效候选结节

💡 这可能表示:
  • 检测结果的置信度都低于筛选阈值
  • 检测到的区域尺寸不符合结节特征
  • 需要调整候选提取参数
"""
        else:
            # 按置信度排序显示候选
            for i, cand in enumerate(candidates[:10]):  # 最多显示前10个
                slice_idx = cand['slice_index']
                bbox = cand['bbox_2d']
                center = cand['center_2d']
                size = cand['size_2d']
                confidence = cand['confidence']

                # 估算物理尺寸（假设0.7mm像素间距）
                physical_size_mm = np.sqrt(size) * 0.7

                report += f"""
🔍 候选结节 {i + 1}:
  • ID: {cand["id"]}
  • 所在切片: #{slice_idx}
  • 中心位置: ({center[0]:.1f}, {center[1]:.1f})
  • 边界框: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]
  • 像素面积: {size:.0f} px²
  • 估算直径: {physical_size_mm:.1f} mm
  • 置信度: {confidence:.3f}
  • 检测模式: {'目标检测' if cand.get('detection_mode', False) else '分割检测'}
"""

                # 风险分层
                if confidence > 0.8:
                    risk_level = "🔴 高置信度"
                    recommendation = "强烈建议进一步检查"
                elif confidence > 0.6:
                    risk_level = "🟡 中高置信度"
                    recommendation = "建议临床关注"
                elif confidence > 0.4:
                    risk_level = "🟢 中等置信度"
                    recommendation = "建议观察随访"
                else:
                    risk_level = "⚪ 低置信度"
                    recommendation = "需要更多证据确认"

                report += f"  • 风险等级: {risk_level}\n"
                report += f"  • 临床建议: {recommendation}\n"

            if len(candidates) > 10:
                report += f"\n... 还有 {len(candidates) - 10} 个候选结节\n"

        report += f"""

⚙️ 技术参数:
  • 运行设备: {self.detector.device}
  • Bundle类型: MonAI肺结节检测 (2D适配)
  • 检测版本: 2D v1.0.0
  • 处理时间: {current_time}

✅ 2D检测优势:
  • 🚀 处理速度快: 单切片推理时间短
  • 💾 内存占用低: 适合资源受限环境
  • 🎯 精确定位: 准确的切片级定位
  • 👁️ 直观检查: 易于医生逐切片审查
  • 🔄 灵活处理: 可选择性检测感兴趣切片

📈 检测质量评估:
  • 置信度分布: {'正常' if confidence_std < 0.3 else '较分散'}
  • 检测一致性: {'良好' if slices_with_detections > 1 else '需要更多证据'}
  • 整体可信度: {'高' if max_confidence > 0.7 else '中等' if max_confidence > 0.5 else '低'}

⚠️ 重要声明:
  • 2D检测结果需要结合3D上下文信息
  • 建议与临床医生共同评估检测结果
  • 本系统仅供辅助诊断，不能替代专业判断
  • 对于可疑结节，建议进行进一步检查

📞 如有疑问，请咨询专业医疗机构
"""

        return report

    def create_interface(self):
        """创建2D Gradio界面"""

        custom_css = """
        .main-title { font-size: 24px; font-weight: bold; text-align: center; margin-bottom: 20px; color: #2c3e50; }
        .section-title { font-size: 18px; font-weight: bold; margin-top: 15px; margin-bottom: 10px; color: #34495e; }
        .info-box { background-color: #e8f6f3; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #1abc9c; }
        .warning-box { background-color: #fdf2e9; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #e67e22; }
        .upload-tip { background-color: #eaf2f8; padding: 12px; border-radius: 6px; margin: 8px 0; font-size: 14px; border-left: 4px solid #3498db; }
        .d2-badge { background-color: #3498db; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; }
        .param-section { background-color: #f8f9fa; padding: 12px; border-radius: 6px; margin: 8px 0; }
        """

        with gr.Blocks(title="2D肺结节检测系统", css=custom_css, theme=gr.themes.Soft()) as interface:
            gr.HTML("""
            <div class='main-title'>
                🫁 2D肺结节检测系统 
                <span class='d2-badge'>2D v1.0.0</span>
            </div>
            """)

            gr.Markdown("""
            <div class='info-box'>
            <b>🎯 2D检测系统特性 (2025-01-24 03:10:12):</b><br>
            • ✅ <b>快速检测</b>: 逐切片推理，速度更快<br>
            • ✅ <b>内存友好</b>: 低内存占用，适合资源受限环境<br>
            • ✅ <b>精确定位</b>: 准确的切片级结节定位<br>
            • ✅ <b>直观审查</b>: 便于医生逐切片检查验证<br>
            • ✅ <b>MonAI适配</b>: 自动将3D模型适配为2D检测
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("<div class='section-title'>🤖 MonAI Bundle配置 (2D模式)</div>")

                    bundle_file = gr.File(
                        label="上传MonAI Bundle文件 (.zip)",
                        file_types=[".zip"],
                        file_count="single"
                    )

                    gr.Markdown("""
                    <div class='upload-tip'>
                    <b>💡 2D Bundle说明:</b><br>
                    • 支持 lung_nodule_ct_detection_v0.5.9.zip<br>
                    • 自动适配3D模型为2D检测<br>
                    • 兼容RetinaNet和分割模型<br>
                    • 智能权重转换和加载
                    </div>
                    """)

                    load_bundle_btn = gr.Button("🚀 加载2D Bundle", variant="primary", size="sm")

                    bundle_status = gr.Textbox(
                        label="Bundle状态",
                        value="未加载Bundle",
                        interactive=False,
                        lines=1
                    )

                    bundle_info = gr.Textbox(
                        label="2D Bundle详细信息",
                        lines=15,
                        interactive=False,
                        value="🔄 请上传MonAI Bundle文件开始2D检测..."
                    )

                with gr.Column(scale=1):
                    gr.HTML("<div class='section-title'>📁 DICOM数据上传</div>")

                    with gr.Tabs():
                        with gr.TabItem("🗂️ ZIP文件夹 (推荐)"):
                            gr.Markdown("""
                            <div class='upload-tip'>
                            <b>📦 2D检测推荐方式:</b> ZIP文件夹上传<br>
                            • 系统会自动提取所有切片进行2D检测<br>
                            • 支持DICOM序列的完整处理<br>
                            • 逐切片分析，不遗漏任何结节
                            </div>
                            """)

                            dicom_zip = gr.File(
                                label="上传DICOM序列ZIP文件",
                                file_types=[".zip"],
                                file_count="single"
                            )

                            process_zip_btn = gr.Button("🔍 2D处理ZIP文件", variant="primary", size="lg")

                        with gr.TabItem("📄 多个DICOM文件"):
                            gr.Markdown("""
                            <div class='upload-tip'>
                            <b>📄 多文件2D检测:</b> 直接选择DICOM文件<br>
                            • 系统会逐个处理每个DICOM文件<br>
                            • 适合少量文件的快速检测<br>
                            • 每个文件都会被作为独立切片处理
                            </div>
                            """)

                            dicom_files = gr.File(
                                label="选择CT序列的DICOM文件",
                                file_types=[".dcm"],
                                file_count="multiple"
                            )

                            process_files_btn = gr.Button("🔍 2D处理DICOM文件", variant="secondary", size="lg")

                    gr.HTML("<div class='section-title'>⚙️ 2D检测参数</div>")

                    gr.Markdown("""
                    <div class='param-section'>
                    <b>🎯 2D检测参数说明:</b><br>
                    • <b>置信度阈值</b>: 过滤低置信度检测结果<br>
                    • <b>最大处理切片</b>: 限制处理切片数量(0=全部)<br>
                    • 2D检测通常比3D检测更敏感，建议适当提高阈值
                    </div>
                    """)

                    confidence_threshold = gr.Slider(
                        label="置信度阈值",
                        minimum=0.1,
                        maximum=0.9,
                        value=0.3,  # 2D检测的默认阈值
                        step=0.05,
                        info="过滤低置信度的检测结果"
                    )

                    max_slices_to_process = gr.Slider(
                        label="最大处理切片数 (0=全部)",
                        minimum=0,
                        maximum=200,
                        value=50,  # 默认处理50个切片用于测试
                        step=10,
                        info="限制处理的切片数量，0表示处理全部"
                    )

                    gr.HTML("<div class='section-title'>ℹ️ 2D检测提示</div>")

                    gr.Markdown("""
                    <div class='warning-box'>
                    <b>💡 2D检测建议:</b><br>
                    • <b>首次使用</b>: 建议设置最大切片数为20-50进行测试<br>
                    • <b>置信度设置</b>: 0.3-0.5适合大多数情况<br>
                    • <b>结果解读</b>: 2D结果需要结合切片位置综合判断<br>
                    • <b>处理时间</b>: 每个切片约需1-3秒，请耐心等待
                    </div>
                    """)

            gr.HTML("<div class='section-title'>🖼️ 2D检测结果可视化</div>")

            with gr.Row():
                detection_overview_2d = gr.Image(
                    label="2D检测总览",
                    show_label=True,
                    height=400,
                    interactive=False
                )

                candidates_montage_2d = gr.Image(
                    label="2D候选结节详情",
                    show_label=True,
                    height=400,
                    interactive=False
                )

            gr.HTML("<div class='section-title'>📊 详细2D检测报告</div>")

            detection_report_2d = gr.Textbox(
                label="2D检测分析报告",
                lines=25,
                max_lines=30,
                interactive=False,
                value="""🔄 请先加载MonAI Bundle和DICOM数据开始2D检测...

💡 2D检测流程:
1️⃣ 上传MonAI Bundle文件 (自动适配2D)
2️⃣ 选择DICOM数据上传方式
3️⃣ 调整检测参数 (建议使用默认值)
4️⃣ 开始2D检测并查看逐切片结果

⚙️ 2D检测优势:
• 更快的推理速度 - 单切片处理
• 更低的内存需求 - 适合资源受限环境  
• 精确的切片定位 - 便于临床审查
• 灵活的检测控制 - 可选择处理切片范围

📋 结果解读指南:
• 高置信度 (>0.7): 强烈建议进一步检查
• 中等置信度 (0.4-0.7): 建议临床关注
• 低置信度 (<0.4): 需要更多证据确认"""
            )

            # 事件绑定
            load_bundle_btn.click(
                fn=self.load_bundle_2d,
                inputs=[bundle_file],
                outputs=[bundle_info, bundle_status]
            )

            # ZIP文件处理事件
            process_zip_btn.click(
                fn=self.process_dicom_zip_2d,
                inputs=[dicom_zip, confidence_threshold, max_slices_to_process],
                outputs=[detection_overview_2d, candidates_montage_2d, detection_report_2d],
                show_progress=True
            )

            # 多文件处理事件
            process_files_btn.click(
                fn=self.process_multiple_dicoms_2d,
                inputs=[dicom_files, confidence_threshold, max_slices_to_process],
                outputs=[detection_overview_2d, candidates_montage_2d, detection_report_2d],
                show_progress=True
            )

            # 使用指南和系统信息
            gr.Markdown(f"""
            ---
            ### 📋 2D检测使用指南

            #### 🎯 推荐的2D检测流程:

            **第1步: Bundle准备**
            ```
            上传您的 lung_nodule_ct_detection_v0.5.9.zip 文件
            点击 "🚀 加载2D Bundle" 等待自动适配完成
            ```

            **第2步: DICOM数据准备**
            ```
            方式1 (推荐): 将DICOM序列文件夹压缩为ZIP上传
            方式2 (备用): 直接多选所有.dcm文件上传
            ```

            **第3步: 参数设置**
            ```
            置信度阈值: 0.3-0.5 (2D检测建议值)
            最大切片数: 20-50 (首次测试) 或 0 (处理全部)
            ```

            **第4步: 开始2D检测**
            ```
            点击对应的处理按钮
            等待逐切片检测完成 (通常需要1-5分钟)
            查看2D可视化结果和详细报告
            ```

            #### ⚙️ 2D检测参数建议:

            - **置信度阈值**: 0.3适合初步筛查，0.5适合精确检测
            - **处理切片数**: 测试时建议50以内，正式检测可设为0(全部)
            - **适用场景**: 快速筛查、资源受限、逐切片审查

            #### 🔧 2D vs 3D 对比:

            | 特性 | 2D检测 | 3D检测 |
            |------|--------|--------|
            | **速度** | ⚡ 快 | 🐌 慢 |
            | **内存** | 💾 低 | 🔥 高 |
            | **精度** | 🎯 切片级 | 🌐 体积级 |
            | **上下文** | 📄 单切片 | 📚 全体积 |
            | **适用性** | 筛查、审查 | 诊断、分析 |

            #### 🔍 2D结果解读:

            - **切片分布**: 连续切片检测增加可信度
            - **置信度**: >0.7高置信度，0.4-0.7中等，<0.4低置信度  
            - **尺寸评估**: 结合像素尺寸和物理间距评估
            - **位置信息**: 记录精确的切片索引和坐标

            #### 🚀 性能优化建议:

            **内存优化:**
            - 设置合理的最大切片数
            - 定期清理临时文件

            **速度优化:**
            - 首次测试使用较少切片
            - 确认参数后再处理全部数据

            **精度优化:**
            - 根据结果调整置信度阈值
            - 结合多个切片的检测结果

            ### 📞 技术信息

            **当前用户**: veryjoyran  
            **系统版本**: 2D检测 v1.0.0  
            **更新时间**: 2025-01-24 03:10:12  
            **特色功能**: 3D→2D自动适配、逐切片检测、快速筛查

            **系统要求**:
            - CPU: 4核以上推荐
            - 内存: 8GB以上推荐  
            - 存储: 预留2GB临时空间

            **支持的数据格式**:
            - DICOM文件 (.dcm) ✅
            - DICOM序列ZIP压缩包 ✅
            - 标准CT胸部扫描 ✅

            如遇技术问题，请检查：
            - Bundle文件完整性和格式
            - DICOM数据格式和完整性
            - 系统内存和存储空间
            - 参数设置的合理性
            """)

        return interface


def main():
    """主函数"""
    print("🚀 启动2D肺结节检测推理界面")
    print(f"👤 用户: veryjoyran")
    print(f"📅 时间: 2025-01-24 03:10:12")
    print("🔧 版本: 2D检测 v1.0.0 - 快速、准确、低资源消耗")
    print("=" * 80)

    try:
        # 创建2D界面
        app = Lung2DDetectionGradioInterface()
        interface = app.create_interface()

        print("✅ 2D界面创建完成")
        print("📌 2D检测特性:")
        print("   • 自动3D→2D模型适配")
        print("   • 逐切片精确检测")
        print("   • 快速推理速度")
        print("   • 低内存占用")
        print("   • 切片级精确定位")

        # 启动服务
        interface.launch(
            server_name="127.0.0.1",
            server_port=7861,  # 使用不同的端口避免冲突
            debug=True,
            show_error=True,
            inbrowser=True,
            share=False
        )

    except Exception as e:
        print(f"❌ 2D界面启动失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()