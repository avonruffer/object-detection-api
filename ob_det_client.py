import requests
import cv2
import numpy as np


def send_image(img_path, server_url):
    # Open image in binary mode
    with open(img_path, "rb") as file:
        response = requests.post(server_url, files={"file": file})

    return response.json()


def main():
    # Define image path and server URL
    IMAGE_PATH = "Bild.png"
    SERVER_URL = "http://127.0.0.1:8000/detect"

    response_data = send_image(IMAGE_PATH, SERVER_URL)

    # Loop through results and print them
    for obj in response_data:
        label = obj["object"]
        confidence = obj["confidence"]
        # DEBUG: print(f"Type of confidence: {type(confidence)}, Value: {confidence}")
        bounding_box = obj["location"]  # [x1, y1, x2, y2]

        print(
            f"Detected {label} with confidence {confidence:.2f} at location {bounding_box}"
        )
    # print(response_data)


if __name__ == "__main__":
    main()
