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
    TEMPORAL_WINDOW = 30
    FIRE_TRIGGER_SECONDS = 10
    CONF_DETECTION_MIN = 0.10

    def __init__(self, video_path: str, model_path: Optional[str] = None, conf_threshold: float = 0.35):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        self.conf_threshold = conf_threshold
        self.running = True
        self.lock = threading.Lock()
        self.cap = None

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
        self.area_history: deque = deque(maxlen=120)

        self.detection_window: deque = deque(maxlen=self.TEMPORAL_WINDOW)
        self.last_temporal_sample_time = 0.0
        self.temporal_status: Dict[str, Any] = {
            'fire_confirmed': False,
            'smoke_confirmed': False,
            'fire_persistence': 0,
            'smoke_persistence': 0,
            'persistence_max': self.TEMPORAL_WINDOW,
            'fire_confidence_stability': 1.0,
            'spatial_trend': 'stable',
        }

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
                area_norm = (area_px / total_pixels) * 100.0

                if cls_id == 0:
                    max_fire_conf = max(max_fire_conf, conf)
                    max_fire_area = max(max_fire_area, area_norm)
                elif cls_id == 1:
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

    def _sample_temporal_window(self) -> None:
        now = time.time()
        if (now - self.last_temporal_sample_time) < 1.0:
            return

        with self.lock:
            fire_conf = self.inference_stats['fire_conf']
            smoke_conf = self.inference_stats['smoke_conf']
            fire_area = self.inference_stats['fire_area']

        self.detection_window.append({
            'fire_conf': fire_conf,
            'smoke_conf': smoke_conf,
            'fire_area': fire_area,
        })
        self.last_temporal_sample_time = now
        self._compute_temporal_status()

    def _compute_temporal_status(self) -> None:
        window = list(self.detection_window)
        if not window:
            return

        fire_detections = [s for s in window if s['fire_conf'] > self.CONF_DETECTION_MIN]
        smoke_detections = [s for s in window if s['smoke_conf'] > self.CONF_DETECTION_MIN]

        fire_persistence = len(fire_detections)
        smoke_persistence = len(smoke_detections)

        fire_confs = [s['fire_conf'] for s in window]
        if len(fire_confs) >= 2:
            mean_conf = sum(fire_confs) / len(fire_confs)
            variance = sum((x - mean_conf) ** 2 for x in fire_confs) / len(fire_confs)
            std_dev = variance ** 0.5
            stability = max(0.0, 1.0 - (std_dev / 0.5))
        else:
            stability = 1.0

        spatial_trend = 'stable'
        fire_areas = [s['fire_area'] for s in window]
        if len(fire_areas) >= 6:
            early_avg = sum(fire_areas[:3]) / 3
            late_avg = sum(fire_areas[-3:]) / 3
            delta = late_avg - early_avg
            if delta > 2.0:
                spatial_trend = 'growing'
            elif delta < -2.0:
                spatial_trend = 'shrinking'

        with self.lock:
            # Current state
            is_confirmed = self.temporal_status.get('fire_confirmed', False)
            
            # TRIGGER: Needs 10 continuous seconds of fire (the most recent 10 samples)
            recent_10 = window[-self.FIRE_TRIGGER_SECONDS:] if len(window) >= self.FIRE_TRIGGER_SECONDS else []
            recent_10_hits = len([s for s in recent_10 if s['fire_conf'] > self.CONF_DETECTION_MIN])
            
            if not is_confirmed and len(window) >= self.FIRE_TRIGGER_SECONDS and recent_10_hits >= self.FIRE_TRIGGER_SECONDS:
                is_confirmed = True
                
            # CLEAR: Needs 30 seconds of ZERO fire (the entire window must have 0 hits)
            elif is_confirmed and fire_persistence == 0 and len(window) >= self.TEMPORAL_WINDOW:
                is_confirmed = False

            # Temporal Risk Score (TRS) calculation
            # TRS = w1*P + w2*C + w3*G
            # P: Persistence (0-100%)
            # C: Confidence Stability (0-100%)
            # G: Growth Trend Penalty (Stable=0, Growing=100%, Shrinking=-50%)
            
            p_score = (fire_persistence / self.TEMPORAL_WINDOW) * 100
            c_score = stability * 100
            
            if spatial_trend == 'growing':
                g_score = 100
            elif spatial_trend == 'shrinking':
                g_score = -50
            else:
                g_score = 0
                
            # Weights: 50% Persistence, 30% Confidence Stability, 20% Growth
            w1, w2, w3 = 0.50, 0.30, 0.20
            
            trs_raw = (w1 * p_score) + (w2 * c_score) + (w3 * g_score)
            trs_final = max(0.0, min(100.0, trs_raw)) # Bound between 0 and 100

            self.temporal_status = {
                'fire_confirmed': is_confirmed,
                'smoke_confirmed': smoke_persistence >= self.FIRE_TRIGGER_SECONDS,
                'fire_persistence': fire_persistence,
                'smoke_persistence': smoke_persistence,
                'persistence_max': self.TEMPORAL_WINDOW,
                'fire_confidence_stability': round(stability, 3),
                'spatial_trend': spatial_trend,
                'risk_score': round(trs_final, 1)
            }

    def _inference_loop(self) -> None:
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

                self._sample_temporal_window()

            except Exception as e:
                print(f"Inference error: {e}")
                time.sleep(0.1)

    def _update_logs(self) -> None:
        current_time = time.time()

        if (current_time - self.last_log_time) > 29.9:
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")

            with self.lock:
                fire = self.inference_stats['fire_conf'] * 100
                smoke = self.inference_stats['smoke_conf'] * 100
                fire_area = self.inference_stats['fire_area']
                smoke_area = self.inference_stats['smoke_area']

            self.area_history.append({
                'time': timestamp,
                'fire_area': fire_area,
                'smoke_area': smoke_area
            })

            has_detection = fire > 10.0 or smoke > 10.0
            log_type = "alert" if has_detection else "normal"
            msg = f"Inference • Fire: {fire:.1f}% • Smoke: {smoke:.1f}%"

            self.logs.appendleft({
                'time': timestamp,
                'msg': msg,
                'type': log_type
            })
            self.last_log_time = current_time

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            return {
                'metrics': self.inference_stats,
                'logs': list(self.logs),
                'growth_history': list(self.area_history),
                'temporal_status': dict(self.temporal_status),
            }

    def stop(self):
        self.running = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        if hasattr(self, 'inference_thread') and self.inference_thread.is_alive():
            self.inference_thread.join(timeout=2.0)

    def start_inference_thread(self):
        if hasattr(self, 'inference_thread') and self.inference_thread.is_alive():
            return

        self.running = True
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()

    def generate_frames(self):
        self.cap = cv2.VideoCapture(str(self.video_path))
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        interval = 1.0 / fps

        if self.model:
            self.start_inference_thread()

        try:
            while self.running:
                start = time.time()
                success, frame = self.cap.read()
                if not success:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
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
            if self.cap is not None:
                self.cap.release()
                self.cap = None
