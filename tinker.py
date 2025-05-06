import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
import time
import os
from threading import Thread
from ultralytics import YOLO
from datetime import datetime
import torch

class YOLOApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Object Detection")
        self.root.geometry("900x700")
        self.root.configure(bg="#0e0b16")  # Dark background for cyberpunk look
        self.root.bind("<Escape>", lambda e: self.root.attributes("-fullscreen", False))

        self.model = None
        self.running = False
        self.video_source = None
        self.vid = None
        self.writer = None
        self.export_logs = tk.BooleanVar(value=True)
        self.export_video = tk.BooleanVar(value=True)
        self.log_file = None
        self.log_entries = []

        self.build_ui()

    def build_ui(self):
        # Create a neon-style menu bar at the top
        controls = ttk.Frame(self.root, padding=10, style="TFrame")
        controls.pack(fill="x", side="top", anchor="n")
        
        # Use custom colors to create the cyberpunk effect
        self.style = ttk.Style(self.root)
        self.style.configure("TButton", padding=6, relief="flat", background="#f20d82", font=("Courier", 10, "bold"))
        self.style.configure("TLabel", background="#0e0b16", foreground="#f20d82", font=("Courier", 12))
        
        # Buttons and checkboxes
        ttk.Button(controls, text="Use Webcam", command=self.use_webcam).grid(row=0, column=0, padx=5)
        ttk.Button(controls, text="Open Video File", command=self.select_video).grid(row=0, column=1, padx=5)
        ttk.Button(controls, text="Start Detection", command=self.start_detection).grid(row=0, column=2, padx=5)
        ttk.Button(controls, text="Stop", command=self.stop_detection).grid(row=0, column=3, padx=5)
        
        ttk.Button(controls, text="Low-End (Pi 3/Pico W)", command=lambda: self.load_model("yolov8n.pt")).grid(row=1, column=0, pady=3)
        ttk.Button(controls, text="Medium-End (CPU)", command=lambda: self.load_model("yolov8m.pt")).grid(row=1, column=1, pady=3)
        ttk.Button(controls, text="High-End (GPU)", command=lambda: self.load_model("yolov8x.pt")).grid(row=1, column=2, pady=3)

        ttk.Checkbutton(controls, text="Export Video", variable=self.export_video).grid(row=2, column=0, columnspan=2, pady=3)
        ttk.Checkbutton(controls, text="Export Log File", variable=self.export_logs).grid(row=2, column=2, columnspan=2, pady=3)

        # Status label with a glowing effect
        self.status_label = ttk.Label(self.root, text="Status: Waiting...", anchor="w", style="TLabel")
        self.status_label.pack(fill="x", padx=10, pady=5)

        # Canvas for the video feed
        self.canvas = tk.Label(self.root, bg="black")
        self.canvas.pack(fill="both", expand=True, padx=10, pady=10)

        self.detect_hardware()

    def detect_hardware(self):
        gpu = torch.cuda.is_available()
        mps = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        device_msg = "GPU Detected (CUDA)" if gpu else "Apple MPS Detected" if mps else "Running on CPU"
        self.status_label.config(text=f"Status: {device_msg}")
        print(f"[INFO] {device_msg}")

    def load_model(self, model_path):
        self.model = YOLO(model_path)
        self.status_label.config(text=f"Loaded model: {model_path}")
        print(f"[INFO] Loaded model: {model_path}")

        if model_path == "yolov8x.pt":
            self.root.attributes("-fullscreen", True)
            self.status_label.config(text="Loaded High-End model: Fullscreen mode enabled.")
        else:
            self.root.attributes("-fullscreen", False)
            self.root.geometry("900x700")

    def use_webcam(self):
        self.video_source = 0
        self.status_label.config(text="Webcam selected.")

    def select_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        if path:
            self.video_source = path
            self.status_label.config(text=f"Selected file: {path}")

    def start_detection(self):
        if self.video_source is None:
            self.status_label.config(text="No video source selected.")
            return

        if self.model is None:
            self.status_label.config(text="No model loaded. Please select a model.")
            return

        if not self.running:
            self.running = True
            self.vid = cv2.VideoCapture(self.video_source)
            self.log_entries = []

            if self.export_video.get():
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = self.vid.get(cv2.CAP_PROP_FPS) or 30
                out_path = self._timestamped_filename("video", ".mp4")
                self.writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
                print(f"[INFO] Writing video to: {out_path}")

            if self.export_logs.get():
                self.log_file = self._timestamped_filename("log", ".txt")
                print(f"[INFO] Writing logs to: {self.log_file}")

            Thread(target=self.detect_loop).start()

    def stop_detection(self):
        self.running = False
        if self.vid:
            self.vid.release()
        if self.writer:
            self.writer.release()
            self.writer = None

        if self.export_logs.get() and self.log_file:
            with open(self.log_file, "w") as f:
                for entry in self.log_entries:
                    f.write(entry + "\n")

        self.status_label.config(text="Detection stopped.")
        print("[INFO] Stopped.")

    def detect_loop(self):
        prev_time = time.time()
        while self.running and self.vid.isOpened():
            ret, frame = self.vid.read()
            if not ret:
                break

            results = self.model(frame, verbose=False)[0]
            annotated = results.plot()

            rgb_image = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            w = self.root.winfo_width()
            h = self.root.winfo_height()
            image = Image.fromarray(rgb_image).resize((w, h), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=image)

            self.canvas.imgtk = imgtk
            self.canvas.configure(image=imgtk)

            now = time.time()
            fps = 1 / (now - prev_time)
            prev_time = now

            detected = []
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = self.model.names[cls_id]
                detected.append(f"{name} ({conf:.2f})")

            debug_text = f"FPS: {fps:.2f} | Detected: {', '.join(detected) if detected else 'None'}"
            self.status_label.config(text=debug_text)

            if self.export_video.get() and self.writer:
                self.writer.write(annotated)

            if self.export_logs.get():
                timestamp = datetime.now().strftime('%H:%M:%S')
                self.log_entries.append(f"[{timestamp}] {debug_text}")

        self.running = False

    def _timestamped_filename(self, prefix, ext):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{ts}{ext}"

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOApp(root)
    root.mainloop()
