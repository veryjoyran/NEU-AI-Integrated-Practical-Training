import torch
from ultralytics import YOLO

if __name__ == '__main__':
    # 打印所有GPU的设备信息
    if torch.cuda.is_available():
        print(f"GPU Count: {torch.cuda.device_count()}")  # 显示GPU数量
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory Allocated: {torch.cuda.memory_allocated(i)} bytes")
            print(f"  Memory Cached: {torch.cuda.memory_reserved(i)} bytes")
    else:
        print("No GPU available, using CPU.")

    # 设置CUDA设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 尝试限制CUDA内存使用，避免内存不足
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()  # 清空缓存
        torch.cuda.set_per_process_memory_fraction(0.8, device=device)  # 限制最大使用内存比例

    # 加载YOLO模型
    model = YOLO(r"D:\PythonProject\NEU-AI-Integrated-Practical-Training\src\pt\yolo11s.pt")  # 加载模型

    # 训练模型
    results = model.train(
        data=r"D:\PythonProject\NEU-AI-Integrated-Practical-Training\LIDC_YOLO_Processed_Dataset\dataset.yaml",
        epochs=100,  # 训练轮数，具体根据需要调整
        imgsz=512,  # 图像尺寸，可以调整为更小的值（例如：512）以减少显存消耗
        batch=16,  # 降低batch size
        device=device,  # 使用正确的设备
        workers=1,  # 降低数据加载的并行度，减少内存消耗
        lr0=0.0001,  # 初始学习率
        lrf=0.01,  # 学习率衰减因子
        momentum=0.937,  # 动量
        weight_decay=0.0005,  # 权重衰减
        warmup_epochs=3,  # 预热训练的轮数
        warmup_momentum=0.8,  # 预热阶段动量
        warmup_bias_lr=0.1,  # 预热阶段的偏置学习率
    )
