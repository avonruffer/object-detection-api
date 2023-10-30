## What is the functionality of this repository?

This repository includes two different fastAPI servers that can be used for externally
processing images for object detection or image segmentation using pre-trained DETR Resnet-50 models.

## Object Detection

The Client opens an image and sends it to the server in a bitstream format.
The server returns an array of detected objects, their scores and their 2D coordinates.

## Image Segmentation

The Client also opens an image and sends it to the server in the same way. The data returned is now a 2D array of integers, where each entry corresponds to one pixel of the original image. Results are printed out by converting the server response to the JSON format.

## How to use

To start the server, we run the following command in the directory of the server script:

```
uvicorn ob_det_server:app --reload
```

"ob_det_server" needs to be replaced with "im_seg_server" for using the image segmentation.

The terminal will tell us that the server is running and under which address it is reachable.
We can now run the client.

```
python3 ob_det_client.py
```

The terminal will print out detected objects, their scores and their locations.

**NOTE:** The conda environments used for development are included in environment.yml files in the repository. Use the "client" environment for running the client and the "server" environment for running the server.
