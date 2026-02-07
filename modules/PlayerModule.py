import cv2
from ultralytics import YOLO
import numpy as np

# --- CONFIGURAÇÃO ---
VIDEO_PATH = "data/raw/video.MP4"

def get_shooter_feet_yolo(video_path=VIDEO_PATH, visualize=True):
    print("Loading YOLOv8-Pose...")
    model = YOLO('models/yolov8n-pose.pt') 

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error to open {video_path}")
        return

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error to read frame.")
        return

    print("Detecting player...")
    # conf=0.5 ffilter low dectections
    results = model.predict(frame, conf=0.5, verbose=False)

    result = results[0]

    if result.keypoints is not None and len(result.keypoints) > 0:
      
        keypoints = result.keypoints.xy[0].cpu().numpy()

        # Índices COCO: 15 (Left Ankle), 16 (Right Ankle)
        left_ankle = keypoints[15]
        right_ankle = keypoints[16]
        
        if np.sum(left_ankle) == 0 or np.sum(right_ankle) == 0:
            print("Aviso: Um dos pés está oculto. Usando a caixa delimitadora como fallback.")
            box = result.boxes.xyxy[0].cpu().numpy() # x1, y1, x2, y2
            center_u = int((box[0] + box[2]) / 2)
            center_v = int(box[3]) 
        else:
            l_u, l_v = int(left_ankle[0]), int(left_ankle[1])
            r_u, r_v = int(right_ankle[0]), int(right_ankle[1])
            
            center_u = int((l_u + r_u) / 2)
            center_v = int((l_v + r_v) / 2)

            print("\n" + "="*30)
            print(f"PÉ ESQUERDO (YOLO): ({l_u}, {l_v})")
            print(f"PÉ DIREITO (YOLO):  ({r_u}, {r_v})")
            print("-" * 30)
            print(f"CENTRO DOS PÉS:     ({center_u}, {center_v})")
            print("="*30 + "\n")

            cv2.circle(frame, (l_u, l_v), 5, (0, 255, 255), -1)
            cv2.circle(frame, (r_u, r_v), 5, (0, 255, 255), -1)

        cv2.circle(frame, (center_u, center_v), 8, (0, 0, 255), -1)
        cv2.putText(frame, f"Pe: {center_u},{center_v}", (center_u+10, center_v), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        h, w = frame.shape[:2]
        view_h = 800
        scale = view_h / h
        view_w = int(w * scale)
        frame_resized = cv2.resize(frame, (view_w, view_h))
        
        if visualize:
            cv2.imshow("YOLO Pose Result", frame_resized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return (center_u, center_v)
    
    else:
        print("Nenhuma pessoa detectada no primeiro frame.")
        return None

if __name__ == "__main__":
    get_shooter_feet_yolo()