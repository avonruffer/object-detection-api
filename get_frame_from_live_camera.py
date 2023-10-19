import cv2
import time

# Start video capture
cap = cv2.VideoCapture(1)  # Use 0 for the default camera; 1: webcam

count_frames = 0
start_time = time.time()

try:
    while True:
        # Read a frame from the video capture
        ret, frame = cap.read()
        if not ret:
            break

        # TODO: Process frame, perform object detection, annotate frame...

        # Determine seconds since program start
        time_elapsed = time.time() - start_time
        fps = count_frames / time_elapsed

        # Annotate live feed with fps
        cv2.putText(
            img=frame,
            text="fps: " + "{0:.2f}".format(fps),
            org=(10, 30),
            fontFace=cv2.FONT_HERSHEY_TRIPLEX,
            fontScale=1,
            color=(0, 255, 0),
            thickness=1,
        )
        count_frames += 1

        # Display the frame
        cv2.imshow("Live Object Detection", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
