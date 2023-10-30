from transformers import DetrFeatureExtractor, DetrForSegmentation
from transformers.image_transforms import rgb_to_id

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
feature_extractor = None


# Load model and processor on startup
@app.on_event("startup")
async def load_model():
    global model, feature_extractor
    feature_extractor = DetrFeatureExtractor.from_pretrained(
        "facebook/detr-resnet-50-panoptic"
    )
    model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")


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
        # TODO Change
        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # use the `post_process_panoptic` method of `DetrFeatureExtractor` to convert to COCO format
        processed_sizes = torch.as_tensor(inputs["pixel_values"].shape[-2:]).unsqueeze(
            0
        )
        result = feature_extractor.post_process_panoptic(outputs, processed_sizes)[0]

        # the segmentation is stored in a special-format png
        panoptic_seg = Image.open(io.BytesIO(result["png_string"]))
        panoptic_seg = np.array(panoptic_seg, dtype=np.uint8)
        # retrieve the ids corresponding to each mask
        panoptic_seg_id = rgb_to_id(panoptic_seg)

        # Convert panoptic_seg_id to make it handeable by fastAPI
        panoptic_seg_id_list = panoptic_seg_id.tolist()

        return panoptic_seg_id_list

    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})
