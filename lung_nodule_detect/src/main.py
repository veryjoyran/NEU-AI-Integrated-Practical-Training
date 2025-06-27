"""
肺结节检测系统总导航页面 - 修正版
Author: veryjoyran
Date: 2025-06-25 15:43:08
Current User: veryjoyran
"""

import gradio as gr
import subprocess
import sys
import os
import time
import threading
from pathlib import Path
from datetime import datetime
import psutil
import signal

class LungNoduleDetectionNavigator:
    """肺结节检测系统总导航器"""

    def __init__(self):
        self.current_user = "veryjoyran"
        self.current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 系统路径配置
        self.systems = {
            "original_yolo": {
                "name": "原生YOLO检测系统",
                "description": "基于原生YOLO的DICOM肺结节检测系统",
                "path": r"D:\python_project\NEU-AI-Integrated-Practical-Training\lung_nodule_detect\src\dcm2yolo\gradio_ui.py",
                "features": [
                    "🏥 DICOM原生支持",
                    "🫁 肺分割预处理",
                    "🎯 YOLO目标检测",
                    "📊 医学影像优化",
                    "🔄 智能预处理流程"
                ],
                "port": 7860,
                "status": "未启动",
                "process": None
            },
            "yolov11_gam": {
                "name": "YOLOv11_GAM增强系统",
                "description": "基于YOLOv11m_GAM_Attention的增强检测系统",
                "path": r"D:\python_project\NEU-AI-Integrated-Practical-Training\lung_nodule_detect\src\yolov11_gam\yolo11_gam_gradio1.py",
                "features": [
                    "🤖 YOLOv11m架构",
                    "🧠 GAM注意力机制",
                    "🏥 DICOM + 图片双支持",
                    "⚡ 实时检测",
                    "📈 高精度识别"
                ],
                "port": 7871,
                "status": "未启动",
                "process": None
            },
            "3d_detection": {
                "name": "3D LUNA16/LIDC检测系统",
                "description": "基于3D体积分析的LUNA16/LIDC兼容检测系统",
                "path": r"D:\python_project\NEU-AI-Integrated-Practical-Training\lung_nodule_detect\src\3D\infer_gradio.py",
                "features": [
                    "🎯 3D体积分析",
                    "📊 LUNA16标准处理",
                    "🔄 LIDC XML注释回退",
                    "🎨 增强可视化",
                    "🏥 医学标准报告"
                ],
                "port": 7869,
                "status": "未启动",
                "process": None
            }
        }

        print(f"🚀 初始化肺结节检测系统总导航")
        print(f"👤 当前用户: {self.current_user}")
        print(f"📅 当前时间: {self.current_time}")
        print(f"🎯 管理 {len(self.systems)} 个检测系统")

    def check_system_paths(self):
        """检查系统路径是否存在"""
        path_status = {}

        for system_id, system_info in self.systems.items():
            path = Path(system_info["path"])
            path_status[system_id] = {
                "exists": path.exists(),
                "path": str(path),
                "name": system_info["name"]
            }

            if path.exists():
                print(f"✅ {system_info['name']}: 路径存在")
            else:
                print(f"❌ {system_info['name']}: 路径不存在 - {path}")

        return path_status

    def launch_system(self, system_id):
        """启动指定系统"""
        try:
            if system_id not in self.systems:
                return f"❌ 未知系统: {system_id}", self._get_system_status()

            system_info = self.systems[system_id]

            # 检查路径是否存在
            if not Path(system_info["path"]).exists():
                return f"❌ 系统文件不存在: {system_info['path']}", self._get_system_status()

            # 检查是否已经启动
            if system_info["process"] is not None and system_info["process"].poll() is None:
                return f"⚠️ {system_info['name']} 已经在运行中", self._get_system_status()

            # 启动系统
            print(f"🚀 启动系统: {system_info['name']}")
            print(f"   路径: {system_info['path']}")
            print(f"   端口: {system_info['port']}")

            # 创建新的Python进程启动系统
            process = subprocess.Popen([
                sys.executable,
                system_info["path"]
            ], cwd=Path(system_info["path"]).parent)

            # 更新系统状态
            self.systems[system_id]["process"] = process
            self.systems[system_id]["status"] = "启动中..."

            # 等待一段时间让系统启动
            time.sleep(2)

            # 检查进程是否还在运行
            if process.poll() is None:
                self.systems[system_id]["status"] = "运行中"
                success_msg = f"""
✅ {system_info['name']} 启动成功！

🌐 访问地址: http://127.0.0.1:{system_info['port']}
🔧 进程ID: {process.pid}
📂 工作目录: {Path(system_info['path']).parent}

💡 系统将在新的浏览器标签页中打开
"""
                return success_msg, self._get_system_status()
            else:
                self.systems[system_id]["status"] = "启动失败"
                return f"❌ {system_info['name']} 启动失败，进程异常退出", self._get_system_status()

        except Exception as e:
            self.systems[system_id]["status"] = "启动失败"
            error_msg = f"❌ 启动 {system_info['name']} 时发生错误: {str(e)}"
            print(error_msg)
            return error_msg, self._get_system_status()

    def stop_system(self, system_id):
        """停止指定系统"""
        try:
            if system_id not in self.systems:
                return f"❌ 未知系统: {system_id}", self._get_system_status()

            system_info = self.systems[system_id]

            if system_info["process"] is None:
                return f"⚠️ {system_info['name']} 未在运行", self._get_system_status()

            # 终止进程
            process = system_info["process"]

            if process.poll() is None:  # 进程还在运行
                print(f"🛑 停止系统: {system_info['name']} (PID: {process.pid})")

                # 尝试温和终止
                try:
                    process.terminate()
                    process.wait(timeout=5)  # 等待5秒
                except subprocess.TimeoutExpired:
                    # 强制终止
                    process.kill()
                    process.wait()

                # 重置状态
                self.systems[system_id]["process"] = None
                self.systems[system_id]["status"] = "已停止"

                return f"✅ {system_info['name']} 已停止", self._get_system_status()
            else:
                self.systems[system_id]["process"] = None
                self.systems[system_id]["status"] = "未启动"
                return f"⚠️ {system_info['name']} 进程已结束", self._get_system_status()

        except Exception as e:
            error_msg = f"❌ 停止 {system_info['name']} 时发生错误: {str(e)}"
            print(error_msg)
            return error_msg, self._get_system_status()

    def stop_all_systems(self):
        """停止所有系统"""
        results = []

        for system_id in self.systems.keys():
            result, _ = self.stop_system(system_id)
            results.append(result)

        final_result = "\n".join(results)
        final_result += f"\n\n🔄 所有系统状态已更新"

        return final_result, self._get_system_status()

    def _get_system_status(self):
        """获取所有系统状态"""
        status_text = f"""
🖥️ 系统状态监控 - {datetime.now().strftime("%H:%M:%S")}

"""

        for system_id, system_info in self.systems.items():
            # 检查进程状态
            if system_info["process"] is not None:
                if system_info["process"].poll() is None:
                    status_icon = "🟢"
                    status = "运行中"
                else:
                    status_icon = "🔴"
                    status = "已停止"
                    system_info["status"] = "已停止"
                    system_info["process"] = None
            else:
                status_icon = "⚪"
                status = "未启动"
                system_info["status"] = "未启动"

            status_text += f"""
{status_icon} {system_info['name']}:
  • 状态: {status}
  • 端口: {system_info['port']}
  • 路径: {Path(system_info['path']).name}
"""

        return status_text

    def get_system_info(self, system_id):
        """获取系统详细信息"""
        if system_id not in self.systems:
            return "❌ 未知系统"

        system_info = self.systems[system_id]
        path_exists = Path(system_info["path"]).exists()

        info_text = f"""
📋 {system_info['name']} 详细信息
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📝 系统描述:
{system_info['description']}

✨ 核心特性:
"""

        for feature in system_info['features']:
            info_text += f"  {feature}\n"

        info_text += f"""
🔧 技术信息:
  • 系统路径: {system_info['path']}
  • 文件状态: {'✅ 存在' if path_exists else '❌ 不存在'}
  • 默认端口: {system_info['port']}
  • 当前状态: {system_info['status']}

🌐 访问方式:
  启动后访问: http://127.0.0.1:{system_info['port']}

💡 使用说明:
  1. 点击"启动系统"按钮启动服务
  2. 系统将自动在新标签页中打开
  3. 完成使用后建议停止系统释放资源
"""

        return info_text

    def create_interface(self):
        """创建导航界面"""

        custom_css = """
        .main-title { 
            font-size: 32px; 
            font-weight: bold; 
            text-align: center; 
            margin-bottom: 30px; 
            color: #2c3e50; 
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        .system-card {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 20px;
            border-radius: 15px;
            margin: 15px 0;
            border-left: 5px solid #007bff;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .system-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }
        .nav-badge { 
            background: linear-gradient(45deg, #007bff, #0056b3);
            color: white; 
            padding: 6px 12px; 
            border-radius: 20px; 
            font-size: 14px; 
            font-weight: bold;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        .control-badge { 
            background: linear-gradient(45deg, #28a745, #1e7e34);
            color: white; 
            padding: 6px 12px; 
            border-radius: 20px; 
            font-size: 14px; 
            font-weight: bold;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        .monitor-badge { 
            background: linear-gradient(45deg, #ffc107, #e0a800);
            color: white; 
            padding: 6px 12px; 
            border-radius: 20px; 
            font-size: 14px; 
            font-weight: bold;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        """

        with gr.Blocks(title="肺结节检测系统总导航", css=custom_css, theme=gr.themes.Soft()) as interface:

            gr.HTML("""
            <div class='main-title'>
                🫁 肺结节检测系统总导航中心
                <br><br>
                <span class='nav-badge'>导航中心</span>
                <span class='control-badge'>系统控制</span>
                <span class='monitor-badge'>状态监控</span>
            </div>
            """)

            gr.Markdown(f"""
            <div style='background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); padding: 25px; border-radius: 15px; margin: 20px 0; border-left: 5px solid #2196f3; box-shadow: 0 4px 8px rgba(0,0,0,0.1);'>
            <b>🎯 系统总导航中心</b><br><br>
            <b>👤 当前用户:</b> {self.current_user}<br>
            <b>📅 访问时间:</b> {self.current_time}<br>
            <b>🖥️ 可用系统:</b> {len(self.systems)} 个专业检测系统<br><br>
            <b>🚀 功能特色:</b><br>
            • ✅ <b>一键启动</b>: 快速启动任意检测系统<br>
            • ✅ <b>实时监控</b>: 监控所有系统运行状态<br>
            • ✅ <b>智能管理</b>: 统一的系统生命周期管理<br>
            • ✅ <b>安全控制</b>: 安全的进程启动和停止<br>
            • ✅ <b>详细信息</b>: 每个系统的完整技术说明
            </div>
            """)

            # 系统选择和控制区域
            with gr.Tabs():

                # 原生YOLO系统
                with gr.TabItem("🏥 原生YOLO检测系统"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.Markdown("""
                            <div class='system-card'>
                            <h3>🏥 原生YOLO检测系统</h3>
                            <p><b>专业医学影像DICOM检测解决方案</b></p>
                            
                            <h4>✨ 核心特性:</h4>
                            <ul>
                                <li>🏥 <b>DICOM原生支持</b>: 直接处理医学影像格式</li>
                                <li>🫁 <b>肺分割预处理</b>: 智能肺区域分割算法</li>
                                <li>🎯 <b>YOLO目标检测</b>: 经典YOLO架构优化</li>
                                <li>📊 <b>医学影像优化</b>: HU值处理和窗宽窗位调整</li>
                                <li>🔄 <b>智能预处理流程</b>: 多策略自适应处理</li>
                            </ul>
                            
                            <h4>🎯 适用场景:</h4>
                            <p>• 医院放射科日常诊断<br>• DICOM标准数据处理<br>• 传统YOLO算法验证</p>
                            </div>
                            """)

                        with gr.Column(scale=1):
                            original_yolo_info = gr.Textbox(
                                label="系统详细信息",
                                lines=15,
                                interactive=False,
                                value="点击查看详情按钮获取系统信息..."
                            )

                            with gr.Row():
                                original_yolo_info_btn = gr.Button("📋 查看详情", variant="secondary")
                                original_yolo_launch_btn = gr.Button("🚀 启动系统", variant="primary")
                                original_yolo_stop_btn = gr.Button("🛑 停止系统", variant="stop")

                # YOLOv11_GAM系统
                with gr.TabItem("🤖 YOLOv11_GAM增强系统"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.Markdown("""
                            <div class='system-card'>
                            <h3>🤖 YOLOv11_GAM增强系统</h3>
                            <p><b>基于最新YOLOv11m架构的GAM注意力增强检测</b></p>
                            
                            <h4>✨ 核心特性:</h4>
                            <ul>
                                <li>🤖 <b>YOLOv11m架构</b>: 最新一代YOLO检测算法</li>
                                <li>🧠 <b>GAM注意力机制</b>: 全局注意力模块增强</li>
                                <li>🏥 <b>DICOM + 图片双支持</b>: 同时支持医学影像和普通图片</li>
                                <li>⚡ <b>实时检测</b>: 毫秒级检测响应</li>
                                <li>📈 <b>高精度识别</b>: 优化的小目标检测能力</li>
                            </ul>
                            
                            <h4>🎯 适用场景:</h4>
                            <p>• 高精度要求的临床检测<br>• 研究级AI算法验证<br>• 实时检测应用场景</p>
                            </div>
                            """)

                        with gr.Column(scale=1):
                            yolov11_gam_info = gr.Textbox(
                                label="系统详细信息",
                                lines=15,
                                interactive=False,
                                value="点击查看详情按钮获取系统信息..."
                            )

                            with gr.Row():
                                yolov11_gam_info_btn = gr.Button("📋 查看详情", variant="secondary")
                                yolov11_gam_launch_btn = gr.Button("🚀 启动系统", variant="primary")
                                yolov11_gam_stop_btn = gr.Button("🛑 停止系统", variant="stop")

                # 3D检测系统
                with gr.TabItem("🎯 3D LUNA16/LIDC检测系统"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.Markdown("""
                            <div class='system-card'>
                            <h3>🎯 3D LUNA16/LIDC检测系统</h3>
                            <p><b>基于3D体积分析的LUNA16/LIDC标准检测系统</b></p>
                            
                            <h4>✨ 核心特性:</h4>
                            <ul>
                                <li>🎯 <b>3D体积分析</b>: 真正的三维体积检测</li>
                                <li>📊 <b>LUNA16标准处理</b>: 符合国际标准的预处理流程</li>
                                <li>🔄 <b>LIDC XML注释回退</b>: AI无检测时显示专家标注</li>
                                <li>🎨 <b>增强可视化</b>: 高分辨率多视图显示</li>
                                <li>🏥 <b>医学标准报告</b>: 临床级别的检测报告</li>
                            </ul>
                            
                            <h4>🎯 适用场景:</h4>
                            <p>• LUNA16挑战数据处理<br>• LIDC数据集研究<br>• 3D医学影像分析</p>
                            </div>
                            """)

                        with gr.Column(scale=1):
                            detection_3d_info = gr.Textbox(
                                label="系统详细信息",
                                lines=15,
                                interactive=False,
                                value="点击查看详情按钮获取系统信息..."
                            )

                            with gr.Row():
                                detection_3d_info_btn = gr.Button("📋 查看详情", variant="secondary")
                                detection_3d_launch_btn = gr.Button("🚀 启动系统", variant="primary")
                                detection_3d_stop_btn = gr.Button("🛑 停止系统", variant="stop")

            # 系统监控和控制区域
            gr.HTML("<h2>🖥️ 系统监控与控制中心</h2>")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("<h3>🎛️ 全局控制</h3>")

                    with gr.Row():
                        refresh_status_btn = gr.Button("🔄 刷新状态", variant="secondary")
                        stop_all_btn = gr.Button("🛑 停止所有系统", variant="stop")

                    global_control_result = gr.Textbox(
                        label="操作结果",
                        lines=8,
                        interactive=False,
                        value="等待操作..."
                    )

                with gr.Column(scale=1):
                    gr.HTML("<h3>📊 实时状态监控</h3>")

                    system_status = gr.Textbox(
                        label="系统状态",
                        lines=12,
                        interactive=False,
                        value=self._get_system_status()
                    )

            # 事件绑定
            def update_status():
                return self._get_system_status()

            # 原生YOLO系统事件
            original_yolo_info_btn.click(
                fn=lambda: self.get_system_info("original_yolo"),
                outputs=original_yolo_info
            )

            original_yolo_launch_btn.click(
                fn=lambda: self.launch_system("original_yolo"),
                outputs=[global_control_result, system_status]
            )

            original_yolo_stop_btn.click(
                fn=lambda: self.stop_system("original_yolo"),
                outputs=[global_control_result, system_status]
            )

            # YOLOv11_GAM系统事件
            yolov11_gam_info_btn.click(
                fn=lambda: self.get_system_info("yolov11_gam"),
                outputs=yolov11_gam_info
            )

            yolov11_gam_launch_btn.click(
                fn=lambda: self.launch_system("yolov11_gam"),
                outputs=[global_control_result, system_status]
            )

            yolov11_gam_stop_btn.click(
                fn=lambda: self.stop_system("yolov11_gam"),
                outputs=[global_control_result, system_status]
            )

            # 3D检测系统事件
            detection_3d_info_btn.click(
                fn=lambda: self.get_system_info("3d_detection"),
                outputs=detection_3d_info
            )

            detection_3d_launch_btn.click(
                fn=lambda: self.launch_system("3d_detection"),
                outputs=[global_control_result, system_status]
            )

            detection_3d_stop_btn.click(
                fn=lambda: self.stop_system("3d_detection"),
                outputs=[global_control_result, system_status]
            )

            # 全局控制事件
            refresh_status_btn.click(
                fn=update_status,
                outputs=system_status
            )

            stop_all_btn.click(
                fn=self.stop_all_systems,
                outputs=[global_control_result, system_status]
            )

            # 添加使用说明
            gr.Markdown(f"""
            ---
            ## 📚 系统使用指南
            
            ### 🚀 启动系统:
            1. **选择系统**: 点击对应的系统标签页
            2. **查看详情**: 点击"查看详情"了解系统信息
            3. **启动服务**: 点击"启动系统"按钮
            4. **自动跳转**: 系统将在新浏览器标签页中打开
            
            ### 🖥️ 系统监控:
            - **实时状态**: 右侧显示所有系统的实时运行状态
            - **刷新状态**: 点击"刷新状态"获取最新信息
            - **批量操作**: 可一键停止所有正在运行的系统
            
            ### 🔧 系统管理:
            - **独立运行**: 每个系统在独立进程中运行，互不干扰
            - **资源控制**: 使用完毕后建议停止系统释放资源
            - **端口配置**: 每个系统使用不同端口，可同时运行
            
            ### 📊 系统对比:
            
            | 系统 | 特色 | 适用场景 | 端口 |
            |------|------|----------|------|
            | 原生YOLO | DICOM专用，传统可靠 | 医院日常诊断 | 7860 |
            | YOLOv11_GAM | 最新算法，高精度 | 研究和高精度需求 | 7871 |
            | 3D LUNA16/LIDC | 3D分析，标准兼容 | 科研和标准数据集 | 7869 |
            
            ### ⚠️ 注意事项:
            - 确保系统文件路径正确且文件存在
            - 首次启动可能需要较长时间加载模型
            - 同时运行多个系统会占用更多内存
            - 建议使用Chrome或Firefox浏览器获得最佳体验
            
            ---
            
            **导航中心版本**: v1.0.1 (修正版)  
            **开发者**: veryjoyran  
            **更新时间**: 2025-06-25 15:43:08  
            **管理系统**: 3个专业肺结节检测系统
            
            **🎯 总导航特色**: 统一管理入口 + 实时状态监控 + 一键启停控制 + 详细系统信息
            
            **🔧 修正内容**: 修复中文引号语法错误，确保系统正常运行
            """)

        return interface

    def cleanup(self):
        """清理资源，停止所有系统"""
        print("🧹 清理导航器资源...")

        for system_id in self.systems.keys():
            try:
                if self.systems[system_id]["process"] is not None:
                    process = self.systems[system_id]["process"]
                    if process.poll() is None:
                        print(f"🛑 停止系统: {self.systems[system_id]['name']}")
                        process.terminate()
                        try:
                            process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            process.kill()
            except Exception as e:
                print(f"⚠️ 清理系统 {system_id} 时出错: {e}")

        print("✅ 导航器资源清理完成")


def main():
    """主函数"""
    print("🚀 启动肺结节检测系统总导航中心 (修正版)")
    print(f"👤 当前用户: veryjoyran")
    print(f"📅 当前时间: 2025-06-25 15:43:08")
    print("🎯 统一管理三个专业检测系统")
    print("🔧 修正版本: 修复中文引号语法错误")
    print("=" * 90)

    try:
        navigator = LungNoduleDetectionNavigator()

        # 检查系统路径
        print("\n📋 检查系统路径状态:")
        path_status = navigator.check_system_paths()

        for system_id, status in path_status.items():
            if status["exists"]:
                print(f"✅ {status['name']}: 准备就绪")
            else:
                print(f"❌ {status['name']}: 路径不存在")
                print(f"   期望路径: {status['path']}")

        # 创建导航界面
        interface = navigator.create_interface()

        print("\n✅ 总导航界面创建成功")
        print("📌 导航中心特性:")
        print("   • 🎛️ 统一控制台 - 一个界面管理所有系统")
        print("   • 🚀 一键启动 - 快速启动任意检测系统")
        print("   • 📊 实时监控 - 监控所有系统运行状态")
        print("   • 🔄 智能管理 - 安全的进程生命周期管理")
        print("   • 📋 详细信息 - 每个系统的完整技术说明")
        print("   • 🛑 批量控制 - 一键停止所有运行系统")
        print("   • 🌐 多端口支持 - 支持同时运行多个系统")
        print("   • 🎨 现代界面 - 标签页设计，用户友好")
        print("   • 🔧 语法修正 - 修复中文引号问题，确保稳定运行")

        print(f"\n🌐 导航中心地址: http://127.0.0.1:7872")
        print("💡 从导航中心可以启动和管理所有检测系统")

        # 注册清理函数
        import atexit
        atexit.register(navigator.cleanup)

        interface.launch(
            server_name="127.0.0.1",
            server_port=7872,  # 导航中心专用端口
            debug=True,
            show_error=True,
            inbrowser=True,
            share=False
        )

    except Exception as e:
        print(f"❌ 导航中心启动失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保资源清理
        try:
            navigator.cleanup()
        except:
            pass


if __name__ == "__main__":
    main()