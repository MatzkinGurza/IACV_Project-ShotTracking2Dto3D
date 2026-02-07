import cv2
from ultralytics import YOLO
import numpy as np

# --- CONFIGURAÇÃO ---
VIDEO_PATH = "C:/Documents/Polimi/IACV/ProjectAssignment/Project/data/raw/video.MP4"

def get_shooter_feet_yolo(video_path=VIDEO_PATH, visualize=True):
    print("Carregando YOLOv8-Pose...")
    # O sufixo '-pose' é a mágica. Ele baixa um modelo treinado para esqueletos.
    model = YOLO('models/yolov8n-pose.pt') 

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro ao abrir {video_path}")
        return

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Erro ao ler frame.")
        return

    print("Detectando jogador...")
    # conf=0.5 filtra detecções fracas
    results = model.predict(frame, conf=0.5, verbose=False)

    # Pega o primeiro resultado
    result = results[0]

    # Verifica se achou alguém com keypoints
    if result.keypoints is not None and len(result.keypoints) > 0:
        # Pega os keypoints da primeira pessoa detectada (índice 0)
        # .xy retorna um tensor com as coordenadas (x, y)
        # shape: (número_pessoas, 17_pontos, 2_coords)
        keypoints = result.keypoints.xy[0].cpu().numpy()

        # Índices COCO: 15 (Left Ankle), 16 (Right Ankle)
        left_ankle = keypoints[15]
        right_ankle = keypoints[16]
        
        # Se os valores forem [0, 0], significa que não detectou aquele ponto específico
        if np.sum(left_ankle) == 0 or np.sum(right_ankle) == 0:
            print("Aviso: Um dos pés está oculto. Usando a caixa delimitadora como fallback.")
            box = result.boxes.xyxy[0].cpu().numpy() # x1, y1, x2, y2
            center_u = int((box[0] + box[2]) / 2)
            center_v = int(box[3]) # Parte de baixo da caixa
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

            # Desenhar para confirmação
            cv2.circle(frame, (l_u, l_v), 5, (0, 255, 255), -1)
            cv2.circle(frame, (r_u, r_v), 5, (0, 255, 255), -1)

        # Desenhar centro
        cv2.circle(frame, (center_u, center_v), 8, (0, 0, 255), -1)
        cv2.putText(frame, f"Pe: {center_u},{center_v}", (center_u+10, center_v), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Mostrar
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