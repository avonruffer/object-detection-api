import requests
import cv2
import numpy as np


def send_image(img_path, server_url):
    # Open image in binary mode
    with open(img_path, "rb") as file:
        response = requests.post(server_url, files={"file": file})

    return response


def main():
    # Define image path and server URL
    IMAGE_PATH = "Bild.png"
    SERVER_URL = "http://127.0.0.1:8000/detect"

    response_data = send_image(
        IMAGE_PATH, SERVER_URL
    )  # response_data is a 2D integer array, that encodes the object type and the instance of the object

    print(response_data.json())


if __name__ == "__main__":
    main()
