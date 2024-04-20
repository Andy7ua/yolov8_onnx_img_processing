import cv2
import os
from main import frame_proc
from img_processing import process_image


def main(model, video_path, filters=False, save=False):
    # Check if the video path exists
    if not os.path.exists(video_path):
        print(f"Error: Video path '{video_path}' does not exist.")
        return

    # Open video file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video '{video_path}'")
        return

    # Prepare to save the video if requested
    if save:
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out_fps = 30  # Adjust this to match the FPS of the input video
        out_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter('predictions/output_video.mp4', fourcc, out_fps, out_size)

    # Read frames from the video
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if no frames are returned

        if filters:
            frame = process_image(frame, comb_num=2)
        detections, frame = frame_proc(model, frame)

        if save:
            out.write(frame)  # Write frame to video file if saving is enabled

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and writer objects
    cap.release()
    if save:
        out.release()
    print("\nVideo processing completed.")


if __name__ == "__main__":
    # Path variables
    model_path = "yolov8n.onnx"
    video_path = "video_1.mov"
    save_video, filters = True, True

    # Load the ONNX model
    model = cv2.dnn.readNetFromONNX(model_path)
    main(model, video_path, filters, save_video)
