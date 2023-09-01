from ultralytics import YOLO

# Load a model
model = YOLO('yolov8x.yaml')  # build a new model from YAML
#model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
#model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data='coco.yaml', epochs=500, imgsz=640)

