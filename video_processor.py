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
    LOG_EMIT_SECONDS = 60.0
    GROWTH_SAMPLE_SECONDS = 30.0
    
    # Shared class-level YOLO model (Singleton pattern) to save RAM
    _shared_model = None
    _fire_cls_id = 0
    _smoke_cls_id = 1
    _model_lock = threading.Lock()
    _inference_lock = threading.Lock() # Prevents YOLO state corruption

    def __init__(self, video_path: str, model_path: Optional[str] = None, conf_threshold: float = 0.35,
                 on_alert_callback=None, camera_name: str = 'Unknown Camera'):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        self.conf_threshold = conf_threshold
        self.on_alert_callback = on_alert_callback  # callable(event: str, cam_name: str) | None
        self.camera_name = camera_name
        self.running = True
        self.lock = threading.Condition()

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
        self.last_growth_sample_time = 0.0
        self.area_history: deque = deque(maxlen=120)

        self.detection_window: deque = deque(maxlen=self.TEMPORAL_WINDOW)
        self.last_temporal_sample_time = 0.0
        self._prev_fire_confirmed = False  # tracks previous state for edge-detection
        self.temporal_status: Dict[str, Any] = {
            'fire_confirmed': False,
            'smoke_confirmed': False,
            'fire_persistence': 0,
            'smoke_persistence': 0,
            'persistence_max': self.TEMPORAL_WINDOW,
            'fire_confidence_stability': 1.0,
            'spatial_trend': 'stable',
            'risk_score': 0.0
        }

        # Lazy load the shared model if not already done
        if model_path and YOLO_AVAILABLE:
            with VideoProcessor._model_lock:
                if VideoProcessor._shared_model is None:
                    try:
                        print(f"Loading shared model: {model_path}")
                        VideoProcessor._shared_model = YOLO(model_path)
                        
                        # Inspect and map classes once
                        if hasattr(VideoProcessor._shared_model, 'names'):
                            names = VideoProcessor._shared_model.names
                            print(f"Shared Model classes: {names}")
                            for idx, name in names.items():
                                if name.lower() == 'fire': VideoProcessor._fire_cls_id = idx
                                if name.lower() == 'smoke': VideoProcessor._smoke_cls_id = idx
                        
                        # Warm up
                        VideoProcessor._shared_model.predict(
                            source=np.zeros((640, 640, 3), dtype=np.uint8), 
                            verbose=False, imgsz=640
                        )
                    except Exception as e:
                        print(f"Shared model load error: {e}")
        
        self.model = VideoProcessor._shared_model
        # Use class-level IDs
        self.fire_cls_id = VideoProcessor._fire_cls_id
        self.smoke_cls_id = VideoProcessor._smoke_cls_id

        self.last_access_time = time.time()
        self.passive_mode = False

        # Shared JPEG buffer
        self.latest_jpeg = None
        self._frame_seq = 0

        # Start worker threads
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        
        self._reader_thread.start()
        self._inference_thread.start()

    @staticmethod
    def _now_ms() -> int:
        """Unix timestamp in milliseconds for client-side local time rendering."""
        return int(time.time() * 1000)

    def _process_detections(self, result: Results) -> Results:
        if result.boxes is None or len(result.boxes) == 0:
            with self.lock:
                self.inference_stats['fire_conf'] = 0.0
                self.inference_stats['smoke_conf'] = 0.0
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

                if cls_id == self.fire_cls_id:
                    max_fire_conf = max(max_fire_conf, conf)
                    max_fire_area = max(max_fire_area, area_norm)
                elif cls_id == self.smoke_cls_id:
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

            # Detect state transitions for alert callbacks
            prev_confirmed = self._prev_fire_confirmed
            self._prev_fire_confirmed = is_confirmed
            transition_event = None
            if not prev_confirmed and is_confirmed:
                transition_event = 'fire_confirmed'
            elif prev_confirmed and not is_confirmed:
                transition_event = 'fire_cleared'

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

        # Fire transition callback — executed outside the lock to avoid deadlocks
        if transition_event and self.on_alert_callback:
            try:
                threading.Thread(
                    target=self.on_alert_callback,
                    args=(transition_event, self.camera_name),
                    daemon=True
                ).start()
            except Exception as e:
                print(f"Alert callback error: {e}")

    def _reader_loop(self) -> None:
        """Background thread 1: Owns VideoCapture, strictly handles frame reading and MJPEG encoding."""
        cap = cv2.VideoCapture(str(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        interval = 1.0 / fps

        try:
            while self.running:
                loop_start = time.time()

                # Power-saving Passive Mode: If not accessed for 30s, slow down significantly
                if (time.time() - self.last_access_time) > 30.0:
                    if not self.passive_mode:
                        ts_ms = self._now_ms()
                        timestamp = datetime.datetime.fromtimestamp(ts_ms / 1000).strftime("%H:%M:%S")
                        self.logs.appendleft({
                            'ts': ts_ms,
                            'time': timestamp,
                            'msg': "[SYSTEM] Idle detected. Entering CPU-Save mode.",
                            'type': 'normal'
                        })
                        self.passive_mode = True
                    time.sleep(5.0) 
                else:
                    if self.passive_mode:
                        ts_ms = self._now_ms()
                        timestamp = datetime.datetime.fromtimestamp(ts_ms / 1000).strftime("%H:%M:%S")
                        self.logs.appendleft({
                            'ts': ts_ms,
                            'time': timestamp,
                            'msg': "[SYSTEM] Activity detected. Resuming Active Mode.",
                            'type': 'normal'
                        })
                        self.passive_mode = False

                success, frame = cap.read()
                if not success:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                # Store raw frame for the inference thread to pick up
                # We use a non-blocking check to update current_frame
                with self.lock:
                    self.current_frame = frame  # Store the object, don't copy yet
                    result = self.latest_result

                # Annotate using the MOST RECENT available AI result (might be a few frames old)
                output = result.plot(img=frame) if result else frame
                ret, buf = cv2.imencode('.jpg', output, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    with self.lock:
                        self.latest_jpeg = buf.tobytes()
                        self._frame_seq += 1
                        self.lock.notify_all()  # Wake up all streaming clients

                self._sample_temporal_window()

                # Pace to video FPS to keep the stream smooth
                elapsed = time.time() - loop_start
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            cap.release()

    def _inference_loop(self) -> None:
        """Background thread 2: Strictly handles YOLO inference on the latest available frame."""
        # Warm up the model (already done in __init__, but good to be safe)
        if not self.model:
            return

        while self.running:
            if self.passive_mode:
                time.sleep(1.0)
                continue

            frame_to_process = None
            with self.lock:
                if self.current_frame is not None:
                    # Capture a copy and clear it so we don't process it twice
                    frame_to_process = self.current_frame
                    self.current_frame = None

            if frame_to_process is not None:
                try:
                    t0 = time.time()
                    
                    # YOLO.predict is NOT thread-safe on a single shared instance!
                    with VideoProcessor._inference_lock:
                        # Optimization: Use imgsz=320 to significantly reduce CPU load
                        results = self.model.predict(
                            frame_to_process, 
                            conf=self.conf_threshold, 
                            iou=0.45, 
                            imgsz=320, 
                            verbose=False
                        )
                    
                    latency = (time.time() - t0) * 1000
                    
                    if results:
                        # Process and update shared results
                        processed_result = self._process_detections(results[0])
                        with self.lock:
                            self.latest_result = processed_result
                            self.inference_stats['latency'] = int(latency)
                        
                        self._update_logs()
                except Exception as e:
                    print(f"Inference error: {e}")
            
            # Small yield to prevent CPU pinning if inference is somehow instant
            time.sleep(0.01)

    def _update_logs(self) -> None:
        current_time = time.time()
        should_sample_growth = (current_time - self.last_growth_sample_time) >= self.GROWTH_SAMPLE_SECONDS
        should_emit_log = (current_time - self.last_log_time) >= self.LOG_EMIT_SECONDS

        if should_sample_growth or should_emit_log:
            ts_ms = self._now_ms()
            timestamp = datetime.datetime.fromtimestamp(ts_ms / 1000).strftime("%H:%M:%S")

            with self.lock:
                fire = self.inference_stats['fire_conf'] * 100
                smoke = self.inference_stats['smoke_conf'] * 100
                fire_area = self.inference_stats['fire_area']
                smoke_area = self.inference_stats['smoke_area']

                if should_sample_growth:
                    self.area_history.append({
                        'ts': ts_ms,
                        'time': timestamp,
                        'fire_area': fire_area,
                        'smoke_area': smoke_area
                    })
                    self.last_growth_sample_time = current_time

                if should_emit_log:
                    has_detection = fire > 10.0 or smoke > 10.0
                    log_type = "alert" if has_detection else "normal"
                    msg = f"Inference - Fire: {fire:.1f}% - Smoke: {smoke:.1f}%"

                    self.logs.appendleft({
                        'ts': ts_ms,
                        'time': timestamp,
                        'msg': msg,
                        'type': log_type
                    })
                    self.last_log_time = current_time

    def get_stats(self) -> Dict[str, Any]:
        self.last_access_time = time.time() # Keeps it alive
        with self.lock:
            # Return copies to prevent race conditions during JSON serialization
            return {
                'metrics': self.inference_stats.copy(),
                'logs': list(self.logs),
                'growth_history': list(self.area_history),
                'temporal_status': self.temporal_status.copy(),
            }

    def stop(self):
        """Signal the background threads to stop and wait for them."""
        self.running = False
        if hasattr(self, '_reader_thread') and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=2.0)
        if hasattr(self, '_inference_thread') and self._inference_thread.is_alive():
            self._inference_thread.join(timeout=2.0)

    def generate_frames(self):
        """Highly efficient generator — waits for new frames using Condition signaling.
        If no new frame arrives within 1 second, it yields the last known frame anyway 
        to detect if the client (browser) has disconnected, preventing thread leaks."""
        self.last_access_time = time.time()
        last_seq = -1
        while self.running:
            self.last_access_time = time.time() # Update on every frame request
            jpeg = None
            with self.lock:
                waited_iters = 0
                # Wait for a NEW frame sequence number or stop signal
                while self.running and self._frame_seq == last_seq and waited_iters < 10:
                    # Timeout prevents deadlocks if the reader thread stalls
                    self.lock.wait(timeout=0.1)
                    waited_iters += 1
                
                if not self.running:
                    break
                
                # Even if _frame_seq == last_seq, we still yield self.latest_jpeg
                # after 1 second of waiting, to ensure the connection is checked
                jpeg = self.latest_jpeg
                last_seq = self._frame_seq
            
            if jpeg:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
            else:
                time.sleep(0.1)

