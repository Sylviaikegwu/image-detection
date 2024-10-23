import torch
from ultralytics import YOLO

model = YOLO('trainedmodels/yolov5_trained_final.pt')  
model.to('cpu')  
torch.set_grad_enabled(False)  

def detect_yolo(image):
    try:
        results = model(image)  
        return results  
    except Exception as e:
        raise RuntimeError("Error during YOLOv5 detection: " + str(e))
