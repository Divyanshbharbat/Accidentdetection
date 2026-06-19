import time
import math
import numpy as np
import cv2
from ultralytics import YOLO

class AccidentDetector:
    def __init__(self, model_path="yolov8n.pt", overlap_threshold=0.45, speed_drop_ratio=0.6, speed_threshold=3.0, alert_cooldown=5.0):
        """
        Args:
            model_path (str): Path to YOLOv8 weights.
            overlap_threshold (float): Minimum Intersection over Minimum Area (IoM) to consider an overlap/collision.
            speed_drop_ratio (float): Percentage drop in speed to qualify as a sudden deceleration (e.g. 0.6 = 60%).
            speed_threshold (float): Minimum speed (pixels/frame) a vehicle must have before collision to be considered active.
            alert_cooldown (float): Cooldown time (seconds) to prevent repeat triggers for the same vehicle pair.
        """
        self.model = YOLO(model_path)
        self.overlap_threshold = overlap_threshold
        self.speed_drop_ratio = speed_drop_ratio
        self.speed_threshold = speed_threshold
        self.alert_cooldown = alert_cooldown
        
        # Track history format: 
        # { track_id: { "centroids": [], "bboxes": [], "speeds": [], "frames": [], "classes": [] } }
        self.track_history = {}
        
        # Triggered accidents format:
        # { (id1, id2): timestamp }  -- order-independent keys (id1 < id2)
        self.triggered_accidents = {}
        
        # Accident logs
        self.accident_events = []
        self.accident_count = 0
        self.frame_counter = 0
        
        # Track vehicle COCO classes: car (2), motorcycle (3), bus (5), truck (7)
        # If it's a custom model, we might want to track other indices, so we'll inspect model names
        self.vehicle_classes = [2, 3, 5, 7]
        self._update_vehicle_classes()
        
    def _update_vehicle_classes(self):
        """Auto-detect vehicle class indices from the model if they are custom named."""
        try:
            names = self.model.names
            custom_classes = []
            vehicle_keywords = ["car", "bike", "motorcycle", "truck", "bus", "vehicle", "van", "suv"]
            for idx, name in names.items():
                if any(kw in name.lower() for kw in vehicle_keywords):
                    custom_classes.append(idx)
            if custom_classes:
                self.vehicle_classes = custom_classes
        except Exception as e:
            print(f"Error reading model classes: {e}. Defaulting to COCO classes.")

    def clear(self):
        """Reset history."""
        self.track_history.clear()
        self.triggered_accidents.clear()
        self.accident_events.clear()
        self.accident_count = 0
        self.frame_counter = 0

    def calculate_overlap(self, boxA, boxB):
        """
        Calculate Intersection over Minimum Area (IoM).
        IoM is more robust than IoU for collisions since vehicles have very different sizes 
        (e.g., motorcycle colliding with a truck).
        """
        # box format: [x1, y1, x2, y2]
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
        if interArea == 0.0:
            return 0.0
            
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        min_area = min(areaA, areaB)
        if min_area == 0.0:
            return 0.0
            
        return interArea / min_area

    def track_and_detect(self, frame, conf_threshold=0.25):
        """
        Processes a single video frame.
        - Detects and tracks vehicles (for detection models) or classifies frame (for classification models).
        - Calculates speed vectors and deceleration (detection only).
        - Detects collisions and flags new accidents.
        Returns:
            annotated_frame (ndarray): Frame with annotations, speeds, and accident tags.
            accident_detected (bool): True if a NEW accident event was triggered in this frame.
            accident_details (dict): Details of the accident if triggered, else None.
        """
        self.frame_counter += 1
        accident_triggered = False
        triggered_detail = None
        current_time = time.time()
        
        # Clean up stale triggered accidents (cooldown expiry)
        stale_pairs = [pair for pair, t in self.triggered_accidents.items() if current_time - t > self.alert_cooldown]
        for pair in stale_pairs:
            del self.triggered_accidents[pair]
            
        # Check model task type
        if getattr(self.model, "task", "detect") == "classify":
            # Run classification inference
            results = self.model(frame, conf=conf_threshold, verbose=False)
            annotated_frame = frame.copy()
            
            if not results or not results[0].probs:
                return annotated_frame, False, None
                
            probs = results[0].probs
            top1_idx = probs.top1
            top1_conf = probs.top1conf.item()
            class_name = self.model.names[top1_idx]
            
            # Check if predicted class is "Accident"
            is_accident = "accident" in class_name.lower() and "non" not in class_name.lower()
            
            # Draw overlay on frame
            color = (0, 0, 255) if is_accident else (0, 255, 0)
            cv2.rectangle(annotated_frame, (10, 10), (320, 85), (30, 30, 30), -1)
            cv2.rectangle(annotated_frame, (10, 10), (320, 85), color, 1)
            cv2.putText(annotated_frame, f"Class: {class_name}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(annotated_frame, f"Conf: {top1_conf:.2%}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Check cooldown and trigger alert
            if is_accident and top1_conf >= conf_threshold:
                pair_key = ("classify", "accident")
                if pair_key not in self.triggered_accidents:
                    self.triggered_accidents[pair_key] = current_time
                    self.accident_count += 1
                    accident_triggered = True
                    
                    triggered_detail = {
                        "id": self.accident_count,
                        "timestamp": time.strftime("%H:%M:%S"),
                        "vehicles": "Entire Frame Classification",
                        "location": "Live Video Feed",
                        "trigger": f"Classification prediction: {class_name} (Conf: {top1_conf:.2%})"
                    }
                    self.accident_events.append(triggered_detail)
                    
            # Draw red warning banner on top if there is any active accident
            if self.triggered_accidents:
                cv2.rectangle(annotated_frame, (0, 0), (frame.shape[1], 45), (0, 0, 255), -1)
                banner_text = "WARNING: ROAD ACCIDENT DETECTED!"
                (w, h), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.putText(annotated_frame, banner_text, (int((frame.shape[1] - w) / 2), 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                            
            return annotated_frame, accident_triggered, triggered_detail

        # Run YOLOv8 built-in tracking (for object detection models)
        results = self.model.track(frame, persist=True, conf=conf_threshold, verbose=False)
        
        annotated_frame = frame.copy()
        
        if not results or not results[0].boxes or results[0].boxes.id is None:
            # No tracks active in this frame
            return annotated_frame, False, None
            
        boxes = results[0].boxes
        xyxy_list = boxes.xyxy.cpu().numpy()
        id_list = boxes.id.cpu().numpy().astype(int)
        cls_list = boxes.cls.cpu().numpy().astype(int)
        conf_list = boxes.conf.cpu().numpy()
        
        current_frame_tracks = {}
        
        # 1. Process active tracks in current frame
        for i, track_id in enumerate(id_list):
            cls_id = cls_list[i]
            # Track only configured vehicle classes
            if cls_id not in self.vehicle_classes:
                continue
                
            box = xyxy_list[i]
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            cls_name = self.model.names[cls_id]
            
            # Initialize track history if new
            if track_id not in self.track_history:
                self.track_history[track_id] = {
                    "centroids": [],
                    "bboxes": [],
                    "speeds": [],
                    "frames": [],
                    "classes": []
                }
                
            hist = self.track_history[track_id]
            hist["centroids"].append((cx, cy))
            hist["bboxes"].append(box)
            hist["frames"].append(self.frame_counter)
            hist["classes"].append(cls_name)
            
            # Calculate speed (pixels / frame)
            speed = 0.0
            if len(hist["centroids"]) > 1:
                prev_cx, prev_cy = hist["centroids"][-2]
                prev_frame = hist["frames"][-2]
                frame_diff = self.frame_counter - prev_frame
                if frame_diff > 0:
                    dist = math.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                    speed = dist / frame_diff
            
            # Keep history sizes capped to avoid memory bloating
            hist["speeds"].append(speed)
            if len(hist["centroids"]) > 30:
                hist["centroids"].pop(0)
                hist["bboxes"].pop(0)
                hist["speeds"].pop(0)
                hist["frames"].pop(0)
                hist["classes"].pop(0)
                
            current_frame_tracks[track_id] = {
                "box": box,
                "centroid": (cx, cy),
                "speed": speed,
                "class_name": cls_name,
                "history": hist
            }
            
        # 2. Collision Analysis between pairs of vehicles active in this frame
        active_ids = list(current_frame_tracks.keys())
        for idxA in range(len(active_ids)):
            for idxB in range(idxA + 1, len(active_ids)):
                idA = active_ids[idxA]
                idB = active_ids[idxB]
                
                vehA = current_frame_tracks[idA]
                vehB = current_frame_tracks[idB]
                
                # Check bounding box overlap
                overlap = self.calculate_overlap(vehA["box"], vehB["box"])
                if overlap > self.overlap_threshold:
                    # Check for sudden speed drop in either vehicle
                    # We compute average speed before the current overlap (e.g. frames -10 to -3)
                    speed_drop_detected = False
                    triggering_id = None
                    
                    for v_id, veh in [(idA, vehA), (idB, vehB)]:
                        speeds = veh["history"]["speeds"]
                        if len(speeds) >= 5:
                            # Speed before collision: average of early history (up to last 3 frames)
                            pre_speeds = speeds[:-3]
                            # Current speed: average of the latest 3 frames
                            post_speeds = speeds[-3:]
                            
                            avg_speed_before = np.mean(pre_speeds) if pre_speeds else 0.0
                            avg_speed_current = np.mean(post_speeds)
                            
                            # If it was moving, and now speed dropped significantly
                            if avg_speed_before > self.speed_threshold:
                                drop = (avg_speed_before - avg_speed_current) / avg_speed_before
                                if drop > self.speed_drop_ratio:
                                    speed_drop_detected = True
                                    triggering_id = v_id
                                    break
                                    
                    # If overlap exists and speed drops suddenly
                    if speed_drop_detected:
                        pair_key = (min(idA, idB), max(idA, idB))
                        
                        # Trigger alert only if not already active in cooldown
                        if pair_key not in self.triggered_accidents:
                            self.triggered_accidents[pair_key] = current_time
                            self.accident_count += 1
                            accident_triggered = True
                            
                            triggering_class = current_frame_tracks[triggering_id]["class_name"] if triggering_id else "vehicle"
                            
                            triggered_detail = {
                                "id": self.accident_count,
                                "timestamp": time.strftime("%H:%M:%S"),
                                "vehicles": f"{vehA['class_name']} #{idA} and {vehB['class_name']} #{idB}",
                                "location": f"Overlap: {overlap:.2f}",
                                "trigger": f"Sudden deceleration of {triggering_class} #{triggering_id}"
                            }
                            self.accident_events.append(triggered_detail)
                            
        # 3. Draw bounding boxes and speeds on frame
        for track_id, veh in current_frame_tracks.items():
            box = veh["box"]
            x1, y1, x2, y2 = map(int, box)
            class_name = veh["class_name"]
            speed = veh["speed"]
            
            # Check if this vehicle is involved in any active accident
            is_in_accident = False
            for (id1, id2) in self.triggered_accidents.keys():
                if track_id == id1 or track_id == id2:
                    is_in_accident = True
                    break
            
            # Choose color: Red for accident-involved, Green for normal
            color = (0, 0, 255) if is_in_accident else (0, 255, 0)
            thickness = 3 if is_in_accident else 2
            
            # Draw bbox
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"{class_name} #{track_id} | Speed: {speed:.1f}px/f"
            if is_in_accident:
                label += " [COLLISION]"
                
            # Text background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        # Draw red warning banner on top if there is any active accident
        if self.triggered_accidents:
            # Red banner background
            cv2.rectangle(annotated_frame, (0, 0), (frame.shape[1], 45), (0, 0, 255), -1)
            banner_text = "WARNING: ROAD ACCIDENT DETECTED!"
            (w, h), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.putText(annotated_frame, banner_text, (int((frame.shape[1] - w) / 2), 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        
        return annotated_frame, accident_triggered, triggered_detail
