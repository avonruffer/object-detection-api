from transformers import DetrImageProcessor, DetrForObjectDetection
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import io
from PIL import Image
import json
import numpy as np
import cv2
import torch


app = FastAPI()

# Global variables for model and processor
model = None
processor = None


# Load model and processor on startup
@app.on_event("startup")
async def load_model():
    global model, processor
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")


# Detect objects on frame
# Input: File as byte-stream
@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    try:
        # Read the received frame to memory
        image_data = await file.read()

        # Convert image back to np array
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert image from np array to PIL image
        # TODO Do in one step?
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Process image and get model output
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        # only keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.9
        )[0]

        detected_objects = []  # Empty list; will be used to return results to client

        # Loop through results
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            # print(type(score), score)  # Let's see what score is
            # print(type(label), label)  # Let's see what label is

            box = [
                round(i) for i in box.tolist()
            ]  # Converts box into a python list and
            # rounds to 2 decimal places

            # Create dict with object information
            obj = {
                "id": label.item(),
                "object": model.config.id2label[label.item()],
                "confidence": float(score.item()),
                "location": box,
            }

            # Append dict to list of results
            detected_objects.append(obj)

        return detected_objects

    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})
