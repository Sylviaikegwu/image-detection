# # import torch
# from ultralytics import YOLO

# def detect_yolo(image):
#     try:
#         model = YOLO('trainedmodels/yolov5_trained_final.pt')  # Your trained model path
#         model.to('cpu')
#         torch.set_grad_enabled(False)
#         results = model(image)
#         return results
#     except Exception as e:
#         raise RuntimeError("Error loading YOLOv5 model: " + str(e))

# import torch
# from ultralytics import YOLO

# model = YOLO('trainedmodels/yolov5_trained_final.pt')  
# model.to('cpu')  
# torch.set_grad_enabled(False)  

# def detect_yolo(image):
#     try:
#         results = model(image)  
#         return results  
#     except Exception as e:
#         raise RuntimeError("Error during YOLOv5 detection: " + str(e))

import logging
from ultralytics import YOLO
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)

def detect_yolo(image):
    try:
        model = YOLO('trainedmodels/yolov5_trained_final.pt')  # Your trained model path
        model.to('cpu')
        torch.set_grad_enabled(False)
        results = model(image)
        logging.info("Detection successful for image: %s", image)
        return results
    except Exception as e:
        logging.error("Error loading YOLOv5 model: %s", str(e))
        raise RuntimeError("Error loading YOLOv5 model: " + str(e))


