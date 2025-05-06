🧠 YOLOv8 Object Detection GUI App

A Python desktop application for real-time object detection using the Ultralytics YOLOv8 model, with a sleek, cyberpunk-style GUI built in Tkinter. Supports webcam or video file input, live annotations, FPS display, export to video, and logging of detection events.
🚀 Features

    🖥️ Cyberpunk-themed GUI with dark mode and neon highlights

    🎥 Live detection via webcam or video files (MP4, AVI, MKV...)

    🧠 Choose between yolov8n, yolov8m, or yolov8x models (Low/Medium/High-end)

    📼 Export annotated detection results to video

    📄 Log detection results to a timestamped text file

    ⚙️ Detects system hardware and auto-adjusts fullscreen for high-end models

    ⏱️ Displays real-time FPS and detected object classes

📦 Requirements

Install dependencies using pip:

pip install ultralytics opencv-python pillow torch

    Note: For GPU acceleration, ensure CUDA is installed and working with PyTorch. macOS M1/M2 users may see MPS support.

🧰 File Structure

yolo_gui_app/
│
├── yolo_gui_app.py        # Main application file
├── yolov8n.pt             # (Optional) Lightweight YOLOv8 model
├── yolov8m.pt             # (Optional) Medium YOLOv8 model
├── yolov8x.pt             # (Optional) High-end YOLOv8 model

🎮 Usage

Run the app with:

python yolo_gui_app.py

💻 Basic Flow

    Choose Input Source:

        Click Use Webcam or Open Video File.

    Select Model Based on System:

        Low-End (Pi 3/Pico W) → yolov8n.pt

        Medium-End (CPU) → yolov8m.pt

        High-End (GPU) → yolov8x.pt (Enables fullscreen)

    Start Detection:

        Click Start Detection.

    Stop Anytime:

        Click Stop to end session and save files.

📂 Output

    Videos saved as: video_YYYYMMDD_HHMMSS.mp4

    Logs saved as: log_YYYYMMDD_HHMMSS.txt

All outputs are saved to the same directory as the script.
🛠️ Customization

You can easily modify:

    Colors/fonts in self.style.configure(...)

    Default model paths (or add auto-download support)

    Logging format or frequency

🧠 Powered By

    Ultralytics YOLOv8

    OpenCV

    Tkinter

    PyTorch



📝 License

This project is for educational and personal use. Modify freely, but check Ultralytics' YOLO license for model usage terms.
