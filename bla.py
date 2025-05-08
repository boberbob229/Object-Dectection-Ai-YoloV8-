import os
import cv2
from ultralytics import YOLO

def main():
    print("YOLOv8 Object Tracker - Terminal Version")
    
    # Ask for video path
    video_path = input("Enter the path to your video file (e.g., video.mp4): ").strip()

    # Validate file
    if not os.path.isfile(video_path):
        print("Error: File not found.")
        return

    print("Running YOLOv8 Tracker...")

    try:
        model = YOLO("yolov8n.pt")  # You can change to yolov8s.pt or your custom model
        model.track(
            source=video_path,
            show=True,
            save=True,
            tracker="bytetrack.yaml",
            persist=True
        )
        print("Tracking complete! Output saved in 'runs/track'")
    except Exception as e:
        print("Tracking failed.")
        print("Error:", str(e))

if __name__ == "__main__":
    main()

