import os
from ultralytics import YOLO

# 设置环境变量以避免 OpenMP 冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def main():
    # Load a model


   # model = YOLO('D:/WJAprogram/YOLOv8/ultralytics-main/ultralytics-main/garbage.yaml')  # build a new model from YAML
   # model = YOLO('D:/WJAprogram/YOLOv8/ultralytics-main/ultralytics-main/yolov8n.pt')  # load a pretrained model (recommended for training)
    model = YOLO('garbage.yaml').load('D:/WJAprogram/YOLOv8/ultralytics-main/ultralytics-main/yolov8n.pt')  # build from YAML and transfer weights

    # Train the model
    model.train(data='D:/WJAprogram/YOLOv8/ultralytics-main/ultralytics-main/garbage.yaml', epochs=3, imgsz=1024)

if __name__ == '__main__':
    main()
