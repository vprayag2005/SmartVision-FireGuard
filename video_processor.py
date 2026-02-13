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
            'latency': 0,
            'model': 'v9 edge'
        }
        self.last_log_time = 0.0

        if model_path and YOLO_AVAILABLE:
            try:
                self.model = YOLO(model_path)
                print(f"Model loaded: {model_path}")
                self.model.predict(source=np.zeros((640, 640, 3), dtype=np.uint8), verbose=False, imgsz=640)
            except Exception as e:
                print(f"Model load error: {e}")

    def _process_detections(self, result: Results) -> Results:
        if result.boxes is None or len(result.boxes) == 0:
            return result

        final_boxes = []
        boxes_by_class: Dict[int, List[List[float]]] = {}

        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            data = box.xyxy[0].cpu().numpy().tolist() + [box.conf[0].item()]
            boxes_by_class.setdefault(cls_id, []).append(data)

        max_fire_conf = 0.0
        max_smoke_conf = 0.0

        for cls_id, boxes in boxes_by_class.items():
            merged_clusters = merge_boxes(boxes)

            if merged_clusters:
                largest = max(merged_clusters, key=lambda c: (c[2] - c[0]) * (c[3] - c[1]))
                final_boxes.append(largest + [float(cls_id)])

                conf = largest[4]
                if cls_id == 0:
                    max_fire_conf = max(max_fire_conf, conf)
                elif cls_id == 1:
                    max_smoke_conf = max(max_smoke_conf, conf)

        with self.lock:
            self.inference_stats['fire_conf'] = float(max_fire_conf)
            self.inference_stats['smoke_conf'] = float(max_smoke_conf)

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
        if (current_time - self.last_log_time) > 4.9:
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            fire = self.inference_stats['fire_conf'] * 100
            smoke = self.inference_stats['smoke_conf'] * 100
            
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
                'logs': list(self.logs)
            }

    def generate_frames(self):
        cap = cv2.VideoCapture(str(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        interval = 1.0 / fps

        if self.model:
            threading.Thread(target=self._inference_loop, daemon=True).start()

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

                output = result.plot(img=frame) if result else frame
                
                ret, buffer = cv2.imencode('.jpg', output, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

                elapsed = time.time() - start
                if interval > elapsed:
                    time.sleep(interval - elapsed)
        finally:
            self.running = False
            cap.release()
