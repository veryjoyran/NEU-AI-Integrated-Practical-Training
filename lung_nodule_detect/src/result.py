import cv2
from ultralytics import YOLO

# 指定本地模型权重路径
local_weight_path = r'D:\PythonProject\NEU-AI-Integrated-Practical-Training\src\runs\detect\train2\weights\best.pt'

# 加载本地YOLOv11模型
model = YOLO(local_weight_path)

# 读取图片
img_path = r'D:\PythonProject\NEU-AI-Integrated-Practical-Training\LIDC_YOLO_Processed_Dataset\images\test\LIDC-IDRI-0001_29880613_000000_26115123_IL057_127364_CT_5637.png'
img = cv2.imread(img_path)
assert img is not None, f"Image {img_path} not found!"

print("开始推理...")

# 推理
results = model(img)

# 获取第一个结果
result = results[0]

# 在原图上绘制检测结果
annotated_img = result.plot()

# 显示结果
cv2.imshow('YOLOv11 Local Inference', annotated_img)
print("按任意键关闭窗口...")
cv2.waitKey(0)
cv2.destroyAllWindows()

# 可选：保存结果图片
output_path = r'D:\PythonProject\NEU-AI-Integrated-Practical-Training\inference_result.jpg'
cv2.imwrite(output_path, annotated_img)
print(f"结果已保存到: {output_path}")

# 打印检测结果详情
if len(result.boxes) > 0:
    print(f"\n检测到 {len(result.boxes)} 个目标:")
    for i, box in enumerate(result.boxes):
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]
        print(f"目标 {i+1}: {class_name}, 置信度: {confidence:.2f}")
else:
    print("未检测到任何目标")