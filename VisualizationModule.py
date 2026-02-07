import matplotlib
from CourtModule import CanonicalCourt3D, CanonicalCourt2D
matplotlib.use('Agg') # Backend não-interativo (mais rápido para renderizar vídeo)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import cv2

class DashboardRenderer:
    def __init__(self, court_3d:CanonicalCourt3D, 
                 court_2d:CanonicalCourt2D, 
                 width=1920*2, height=1080):
        self.court_3d = court_3d
        self.court_2d = court_2d
        self.W = width
        self.H = height
        
        # Configuração do Layout
        # Vídeo Principal: Esquerda (Metade da tela)
        self.video_w = int(width * 0.5)
        self.video_h = height
        
        # Gráficos: Direita (Metade da tela, dividida em 2)
        self.plot_w = int(width * 0.5)
        self.plot_h = int(height * 0.5)

        # Inicializa as Figuras do Matplotlib (para não recriar a cada frame)
        self._init_plots()

    def _init_plots(self):
        plt.rcParams['figure.facecolor'] = '#1f3b5c'

        # --- FIGURA 3D (Canto Superior Direito) ---
        self.fig_3d = plt.figure(figsize=(8, 7), dpi=100)
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        self.canvas_3d = FigureCanvas(self.fig_3d)
        
        # Desenhar linhas da quadra 3D (Estáticas)
        self.court_3d.plot_court(ax=self.ax_3d, return_ax=True) 
        
        # Ajustes de visualização 3D
        self.ax_3d.view_init(elev=20, azim=-45)
        self.ax_3d.set_title("Reconstrução 3D", color='black')
        self.ax_3d.dist = 10

        # --- FIGURA 2D TOP-DOWN (Canto Inferior Direito) ---
        self.fig_2d = plt.figure(figsize=(8, 7), dpi=100)
        self.ax_2d = self.fig_2d.add_subplot(111)
        self.canvas_2d = FigureCanvas(self.fig_2d)
        
        # Desenhar linhas da quadra 2D
        self.court_2d.plot_court(ax = self.ax_2d, return_ax=True) 
        self.ax_2d.set_title("Vista Superior (Top-Down)", color='black')
        self.ax_2d.set_aspect('equal')
        self.ax_2d.invert_yaxis() # Y cresce para baixo na quadra 2D geralmente

        # Inicializa linhas vazias para a trajetória da bola (para atualizar depois)
        self.line_3d, = self.ax_3d.plot([], [], [], 'o-', color='orange', markersize=4, markevery=5)
        self.line_2d, = self.ax_2d.plot([], [], 'o-', color='orange', markersize=4, markevery=5)
        
        # Ponto atual (bola)
        self.point_3d, = self.ax_3d.plot([], [], [], 'o', color='red', markersize=8)
        self.point_2d, = self.ax_2d.plot([], [], 'o', color='red', markersize=8)

    def render_frame(self, video_frame, trajectory_points_up_to_now):
        """
        Recebe o frame original e a lista de pontos 3D até o momento atual.
        Retorna uma imagem combinada (Dashboard).
        """
        # 1. Preparar Frame de Vídeo (Esquerda)
        if video_frame is not None:
            resized_video = cv2.resize(video_frame, (self.video_w, self.video_h))
        else:
            resized_video = np.zeros((self.video_h, self.video_w, 3), dtype=np.uint8)

        # 2. Atualizar Gráficos (Se houver trajetória)
        if trajectory_points_up_to_now and len(trajectory_points_up_to_now) > 0:
            pts = np.array(trajectory_points_up_to_now)
            xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]
            
            # Atualiza 3D
            self.line_3d.set_data(xs, ys)
            self.line_3d.set_3d_properties(zs)
            self.point_3d.set_data([xs[-1]], [ys[-1]])
            self.point_3d.set_3d_properties([zs[-1]])
            
            # Atualiza 2D
            self.line_2d.set_data(xs, ys)
            self.point_2d.set_data([xs[-1]], [ys[-1]])
        
        # 3. Renderizar Matplotlib para Imagem (FIX PARA MATPLOTLIB RECENTE)
        
        # -> Render 3D
        self.canvas_3d.draw()
        # Pega o buffer como array RGBA (Red, Green, Blue, Alpha)
        img_3d = np.asarray(self.canvas_3d.buffer_rgba())
        # Converte RGBA -> BGR (OpenCV usa BGR)
        img_3d = cv2.cvtColor(img_3d, cv2.COLOR_RGBA2BGR)
        img_3d = cv2.resize(img_3d, (self.plot_w, self.plot_h))
        
        # -> Render 2D
        self.canvas_2d.draw()
        img_2d = np.asarray(self.canvas_2d.buffer_rgba())
        img_2d = cv2.cvtColor(img_2d, cv2.COLOR_RGBA2BGR)
        img_2d = cv2.resize(img_2d, (self.plot_w, self.plot_h))

        # 4. Montar o Dashboard Final
        dashboard = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        
        # Colar Video (Esquerda)
        dashboard[0:self.video_h, 0:self.video_w] = resized_video
        
        # Colar 3D (Direita Superior)
        dashboard[0:self.plot_h, self.video_w:] = img_3d
        
        # Colar 2D (Direita Inferior)
        dashboard[self.plot_h:, self.video_w:] = img_2d
        
        return dashboard

    def close(self):
        plt.close(self.fig_3d)
        plt.close(self.fig_2d)