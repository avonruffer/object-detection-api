from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests
import cv2
import numpy as np


def detect_objects(frame):
    # Convert frame from cv2 format (numpy array) to PIL Image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Process image and get model output
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.9
    )[0]

    # Loop through results and draw bounding boxes and labels
    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        box = [round(i) for i in box.tolist()]  # Converts box into a python list and
        # rounds to 2 decimal places

        # Prints detected objects
        print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )

        # Get coordinates
        x, y, xmax, ymax = box

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (xmax, ymax), (23, 230, 210), thickness=2)

        # Draw label
        label_text = f"{model.config.id2label[label.item()]}: {round(score.item(), 3)}"
        cv2.putText(
            frame,
            label_text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (23, 230, 210),
            2,
        )

    return frame


# Load the processor and the model
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Start video capture
cap = cv2.VideoCapture(1)  # 1 - Mac main camera

try:
    while True:
        # Read a frame from video capture
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame, detect object and annotate frame
        annotated_frame = detect_objects(frame)

        # Display frame
        cv2.imshow("Live Object Detection", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
