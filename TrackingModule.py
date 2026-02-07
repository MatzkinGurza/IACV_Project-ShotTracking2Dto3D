import cv2 
import numpy as np
import json
import os
from typing import Literal, Optional, Tuple
import mediapipe as mp
from ultralytics import YOLO
from PlayerModule import get_shooter_feet_yolo


# Classes

class BallTracker:
    def __init__(self, mode: Literal['simple', 'robust']='simple', metadata_path=None):
        self.mode = mode  # 'simple' ou 'robust' (TrackNet/DL)
        self.prev_circle = None
        self.tracks = []
        self.metadata_path = metadata_path

        
        if self.mode == 'robust':
            raise Warning("Robust Tracker is still now yielding satisfactory results")
            print("Aviso: Carregando arquitetura Yolo")
            self.MODEL_SIZE = 'n' # nano
            print(f"Carregando YOLOv8{self.MODEL_SIZE}...")
            self.model = YOLO(f'yolov8{self.MODEL_SIZE}.pt') 
            
        
    def _resolution_scaling(self, calib_h, calib_w, video_path):
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            ret, first_frame = cap.read()
            if ret:
                video_h, video_w = first_frame.shape[:2]
                print(f"\n[TRACKER DIAGNOSTIC]")
                print(f"Real video Resolution: {video_w}x{video_h}")
                scale_x, scale_y = 1.0, 1.0
                print(f"Calibration Resolution Given: {calib_w}x{calib_h}")
                if video_w != calib_w or video_h != calib_h:
                    scale_x = calib_w / video_w
                    scale_y = calib_h / video_h
                    print(f"Mismatch Detected! Aplying Scaling factor: X={scale_x:.3f}, Y={scale_y:.3f}")
                else:
                    print("Resolutions were already in sync... Continuing")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cap.release()
        cv2.destroyAllWindows()
        return scale_x, scale_y


    def _dist(self, p1, p2):
        return (float(p1[0]) - float(p2[0]))**2 + (float(p1[1]) - float(p2[1]))**2

    def _simple_hough_detection(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (17, 17), 0)
        
        # Parâmetros ajustados para basquete
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
            param1=100, param2=30, minRadius=10, maxRadius=100
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            chosen = None
            
            # Heurística de proximidade
            if self.prev_circle is None:
                chosen = circles[0, 0]
            else:
                best_dist = float('inf')
                for c in circles[0, :]:
                    d = self._dist(c[:2], self.prev_circle[:2])
                    if d < best_dist:
                        best_dist = d
                        chosen = c
            return chosen
        return None

    def _robust_yolo_detection(self, frame):
        results = self.model.predict(frame, classes=[32], verbose=False, conf=0.3)
        best_conf = -1
        ball_data = None

        # Processar detecções
        for r in results:
            boxes = r.boxes
            for box in boxes:
                conf = float(box.conf[0])
                if conf > best_conf:
                    # Bounding Box
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Centro e Raio aproximado
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    w_box = x2 - x1
                    h_box = y2 - y1
                    radius = (w_box + h_box) / 4 # Média do diâmetro / 2
                    
                    best_conf = conf
                    ball_data = (cx, cy, radius)
        return ball_data

    def track_video(self, video_path, calibration_resolution:Tuple[int, int],
                    save_to_metadata: bool = False, output_json:Optional[str]=None,
                    visualize: bool = True):
        cap = cv2.VideoCapture(video_path)
        frame_id = 0

        scale_x, scale_y = self._resolution_scaling(calibration_resolution[0],
                                                    calibration_resolution[1],
                                                    video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # Seleciona o método baseado no modo
            if self.mode == 'simple':
                ball_data = self._simple_hough_detection(frame)
            elif self.mode == 'robust':
                ball_data = self._robust_yolo_detection(frame)
            else:
                raise ValueError(f"Unknown Mode: {self.mode}")

            if ball_data is not None:
                raw_x, raw_y, raw_r = ball_data
                x = raw_x * scale_x
                y = raw_y * scale_y
                r = raw_r * scale_x
                self.tracks.append({
                    "frame": frame_id,
                    "x": float(x),
                    "y": float(y),
                    "r": float(r)
                })
                self.prev_circle = ball_data
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_x = int(raw_x + 50)
                text_y = int(raw_y - 50)

                cv2.putText(
                    frame,
                    f"({x:.1f}, {y:.1f}, {r:.1f})",
                    (text_x, text_y),
                    font,
                    1.0,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                    )
                # Feedback visual
                cv2.circle(frame, (int(raw_x), int(raw_y)), int(raw_r), (255, 0, 255), 2)

            if visualize:
                cv2.imshow("Tracking", cv2.resize(frame, (1280, 720)))
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            frame_id += 1

        cap.release()
        cv2.destroyAllWindows()
        if save_to_metadata and self.metadata_path:
            self._save_json(self.metadata_path)
        if output_json: self._save_json(output_json)
        return self.tracks
    
    def get_ball_uv_start(self, video_path, calibration_resolution:Tuple[int, int], visualize=True):
        u,v = get_shooter_feet_yolo(video_path, visualize=visualize)
        scale_u, scale_v = self._resolution_scaling(calibration_resolution[0],
                                                    calibration_resolution[1],
                                                    video_path)

        return u*scale_u, v*scale_v

    def _save_json(self, path):
        if os.path.exists(path):
            with open(path, 'r+') as f:
                data = json.load(f)
                data["tracking"] = self.tracks
                f.seek(0)
                json.dump(data, f, indent=4)
        else:
            with open(path, 'w') as f:
                json.dump(self.tracks, f, indent=2)
        print(f"Tracking salvo em {path}")
