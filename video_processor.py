import threading
import time
import datetime
from collections import deque
from pathlib import Path
from typing import Optional, Dict, List, Any

import cv2
import numpy as np

try:
    import torch
    from ultralytics import YOLO
    from ultralytics.engine.results import Boxes, Results
    YOLO_AVAILABLE = True
except ImportError:
    torch = None
    YOLO = None
    Boxes = None
    Results = None
    YOLO_AVAILABLE = False


def merge_boxes(boxes: List[List[float]]) -> List[List[float]]:
    """Merges overlapping boxes and keeps the union."""
    if not boxes:
        return []

    clusters = []
    
    while boxes:
        current = boxes.pop(0)
        merged = True
        while merged:
            merged = False
            rest = []
            for other in boxes:
                x_min = max(current[0], other[0])
                y_min = max(current[1], other[1])
                x_max = min(current[2], other[2])
                y_max = min(current[3], other[3])

                if x_min < x_max and y_min < y_max:
                    current[0] = min(current[0], other[0])
                    current[1] = min(current[1], other[1])
                    current[2] = max(current[2], other[2])
                    current[3] = max(current[3], other[3])
                    current[4] = max(current[4], other[4])
                    merged = True
                else:
                    rest.append(other)
            boxes = rest
        clusters.append(current)

    return clusters


class VideoProcessor:
    def __init__(self, video_path: str, model_path: Optional[str] = None, conf_threshold: float = 0.30):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        self.conf_threshold = conf_threshold
        self.running = True
        self.lock = threading.Lock()
        
        self.model = None
        self.latest_result: Optional[Results] = None
        self.current_frame: Optional[np.ndarray] = None
        
        self.logs: deque = deque(maxlen=200)
        self.inference_stats: Dict[str, Any] = {
            'fire_conf': 0.0,
            'smoke_conf': 0.0,
            'fire_area': 0.0,
            'smoke_area': 0.0,
            'latency': 0,
            'model': 'v9 edge'
        }
        self.last_log_time = 0.0
        self.area_history: deque = deque(maxlen=120)  # Store last 120 snapshots (1 hour at 30s interval)

        if model_path and YOLO_AVAILABLE:
            try:
                self.model = YOLO(model_path)
                print(f"Model loaded: {model_path}")
                self.model.predict(source=np.zeros((640, 640, 3), dtype=np.uint8), verbose=False, imgsz=640)
            except Exception as e:
                print(f"Model load error: {e}")

    def _process_detections(self, result: Results) -> Results:
        if result.boxes is None or len(result.boxes) == 0:
            with self.lock:
                self.inference_stats['fire_area'] = 0.0
                self.inference_stats['smoke_area'] = 0.0
            return result

        final_boxes = []
        boxes_by_class: Dict[int, List[List[float]]] = {}

        # Get image dimensions for normalization
        img_h, img_w = result.orig_shape
        total_pixels = img_h * img_w

        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            data = box.xyxy[0].cpu().numpy().tolist() + [box.conf[0].item()]
            boxes_by_class.setdefault(cls_id, []).append(data)

        max_fire_conf = 0.0
        max_smoke_conf = 0.0
        max_fire_area = 0.0
        max_smoke_area = 0.0

        for cls_id, boxes in boxes_by_class.items():
            merged_clusters = merge_boxes(boxes)

            if merged_clusters:
                largest = max(merged_clusters, key=lambda c: (c[2] - c[0]) * (c[3] - c[1]))
                final_boxes.append(largest + [float(cls_id)])

                conf = largest[4]
                width = largest[2] - largest[0]
                height = largest[3] - largest[1]
                area_px = width * height
                area_norm = (area_px / total_pixels) * 100.0  # Percentage of screen

                if cls_id == 0: # Fire
                    max_fire_conf = max(max_fire_conf, conf)
                    max_fire_area = max(max_fire_area, area_norm)
                elif cls_id == 1: # Smoke
                    max_smoke_conf = max(max_smoke_conf, conf)
                    max_smoke_area = max(max_smoke_area, area_norm)

        with self.lock:
            self.inference_stats['fire_conf'] = float(max_fire_conf)
            self.inference_stats['smoke_conf'] = float(max_smoke_conf)
            self.inference_stats['fire_area'] = float(max_fire_area)
            self.inference_stats['smoke_area'] = float(max_smoke_area)

        if final_boxes:
            device = result.boxes.xyxy.device
            tensor_data = torch.tensor(final_boxes, device=device)
            result.boxes = Boxes(tensor_data, result.orig_shape)
        
        return result

    def _inference_loop(self) -> None:
        print("Inference thread started")
        while self.running:
            with self.lock:
                frame = self.current_frame.copy() if self.current_frame is not None else None

            if frame is None:
                time.sleep(0.01)
                continue

            try:
                if self.model:
                    start = time.time()
                    results = self.model.predict(frame, conf=self.conf_threshold, iou=0.5, verbose=False)
                    latency = (time.time() - start) * 1000

                    if results:
                        processed = self._process_detections(results[0])
                        with self.lock:
                            self.latest_result = processed
                            self.inference_stats['latency'] = int(latency)
                            self._update_logs()

            except Exception as e:
                print(f"Inference error: {e}")
                time.sleep(0.1)

    def _update_logs(self) -> None:
        current_time = time.time()
        # Update logs every 30 seconds (approx)
        if (current_time - self.last_log_time) > 29.9:
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            fire = self.inference_stats['fire_conf'] * 100
            smoke = self.inference_stats['smoke_conf'] * 100
            
            # Record growth history
            self.area_history.append({
                'time': timestamp,
                'fire_area': self.inference_stats['fire_area'],
                'smoke_area': self.inference_stats['smoke_area']
            })
            
            msg = f"CAM-01 Snapshot • Fire: {fire:.1f}% • Smoke: {smoke:.1f}%"
            self.logs.appendleft({
                'time': timestamp,
                'msg': msg,
                'type': "normal"
            })
            self.last_log_time = current_time

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            return {
                'metrics': self.inference_stats,
                'logs': list(self.logs),
                'growth_history': list(self.area_history)
            }

    def start_inference_thread(self):
        """Starts the inference thread if it's not already running."""
        # Check if thread is alive
        if hasattr(self, 'inference_thread') and self.inference_thread.is_alive():
            return

        self.running = True
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()

    def generate_frames(self):
        cap = cv2.VideoCapture(str(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        interval = 1.0 / fps

        if self.model:
            self.start_inference_thread()

        try:
            while True:
                start = time.time()
                success, frame = cap.read()
                if not success:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                with self.lock:
                    self.current_frame = frame
                    result = self.latest_result

                # Check if result is stale (optional optimization) or just plot whatever we have
                output = result.plot(img=frame) if result else frame
                
                ret, buffer = cv2.imencode('.jpg', output, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

                elapsed = time.time() - start
                if interval > elapsed:
                    time.sleep(interval - elapsed)
        finally:
            # Do NOT stop the inference thread here. 
            # It should keep running or be managed globally.
            # self.running = False 
            cap.release()
