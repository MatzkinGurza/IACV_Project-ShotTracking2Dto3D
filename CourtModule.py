import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple

class CanonicalCourt2D:
    def __init__(self):
        self.COURT_WIDTH = 15.0
        self.HALF_COURT_LENGTH = 14.0
        self.RIM_HEIGHT = 3.05
        
        # Key (Paint)
        self.KEY_WIDTH = 4.9
        self.KEY_LENGTH = 5.8
        
        # Rim & Backboard
        self.RIM_RADIUS = 0.225
        self.BACKBOARD_OFFSET = 1.2
        self.RIM_OFFSET = 1.575         # Distance from baseline to rim center
        self.NO_CHARGE_RADIUS = 1.25
        
        # 3-Point Line
        self.THREE_POINT_RADIUS = 6.75
        self.THREE_POINT_SIDE_OFFSET = 0.90

        # Free Throw
        self.FREE_THROW_RADIUS = 1.8

        # Define specific requested points (0 to 7)
        self.canonical_points = self._define_canonical_points()

    def _define_canonical_points(self):
        """
        Defines the 8 specific intersection points requested.
        Origin (0,0) = RIM CENTER.
        """
        pts = {}

        # Y coordinates relative to Rim (0,0)
        y_base = - self.RIM_OFFSET
        y_ft = self.KEY_LENGTH - self.RIM_OFFSET 

        # X coordinates
        x_key = self.KEY_WIDTH / 2
        x_3pt = (self.COURT_WIDTH / 2) - self.THREE_POINT_SIDE_OFFSET
        z_rim = self.RIM_HEIGHT

        # ----------------------------------------------------
        # MAPPING (Renumbering 0-7)
        # ----------------------------------------------------

        # --- mesmos IDs, mesmos significados ---
        pts[0] = ( -x_3pt, y_base, 0)
        pts[2] = ( x_key, y_base, 0)
        pts[1] = ( -x_key, y_base, 0)
        pts[4] = ( x_key, y_ft, 0)
        pts[3] = ( -x_key, y_ft, 0)

        
        return pts

    def get_canonical_points(self):
        return [self.canonical_points[i] for i in sorted(self.canonical_points.keys())]

    def plot_court(self, add_trajectory: bool = False, 
                   trajectory_points: list = None,
                   return_ax:bool=False,
                   ax=None):
        """
        Draws the court model.
        """
        # Configuração da Figura 3D
        if ax is None:
            # Reduced figure size for a smaller popup
            plt.rcParams['figure.facecolor'] = '#1f3b5c'
            fig, ax = plt.subplots(figsize=(6, 7))
            

        # Styling
        plt.rcParams['figure.facecolor'] = '#1f3b5c'
        ax.set_facecolor('#1f3b5c') 

        LINE_COLOR = 'white'
        LINE_WIDTH = 2
        
        # Geometry helpers for drawing
        y_base = -self.RIM_OFFSET
        y_ft = self.KEY_LENGTH - self.RIM_OFFSET
        y_half = self.HALF_COURT_LENGTH - self.RIM_OFFSET

        # 1. Main Rectangle
        ax.add_patch(patches.Rectangle(
            (-self.COURT_WIDTH/2, y_base), 
            self.COURT_WIDTH, self.HALF_COURT_LENGTH, 
            linewidth=LINE_WIDTH, edgecolor=LINE_COLOR, fill=False
        ))

        # 2. Key Rectangle
        ax.add_patch(patches.Rectangle(
            (-self.KEY_WIDTH/2, y_base), 
            self.KEY_WIDTH, self.KEY_LENGTH, 
            linewidth=LINE_WIDTH, edgecolor=LINE_COLOR, fill=False
        ))

        # 3. Free Throw Circle
        ax.add_patch(patches.Circle(
            (0, y_ft), self.FREE_THROW_RADIUS, 
            linewidth=LINE_WIDTH, edgecolor=LINE_COLOR, fill=False
        ))

        # 4. 3-Point Line
        x_straight = (self.COURT_WIDTH/2) - self.THREE_POINT_SIDE_OFFSET
        dy = math.sqrt(self.THREE_POINT_RADIUS**2 - x_straight**2)
        
        # Straight parts
        ax.plot([-x_straight, -x_straight], [y_base, dy], color=LINE_COLOR, linewidth=LINE_WIDTH)
        ax.plot([x_straight, x_straight], [y_base, dy], color=LINE_COLOR, linewidth=LINE_WIDTH)

        # Arc part
        theta_deg = math.degrees(math.acos(abs(x_straight) / self.THREE_POINT_RADIUS))

        ax.add_patch(patches.Arc(
            (0, 0),
            2*self.THREE_POINT_RADIUS, 2*self.THREE_POINT_RADIUS,
            theta1= theta_deg,
            theta2= 180 - theta_deg,
            linewidth=LINE_WIDTH,
            edgecolor=LINE_COLOR
        ))

        # 5. Restricted Area
        ax.add_patch(patches.Arc(
            (0, 0), 
            2*self.NO_CHARGE_RADIUS, 2*self.NO_CHARGE_RADIUS,
            theta1=0, theta2=180,
            linewidth=LINE_WIDTH, edgecolor=LINE_COLOR
        ))

        # 6. Backboard
        y_backboard = y_base + self.BACKBOARD_OFFSET
        ax.plot([-0.9, 0.9], [y_backboard, y_backboard], color=LINE_COLOR, linewidth=LINE_WIDTH)

        # 7. Rim
        ax.add_patch(patches.Circle(
            (0, 0), self.RIM_RADIUS, 
            linewidth=LINE_WIDTH, edgecolor=LINE_COLOR, fill=False
        ))

        # 8. Plot Requested Points (0-7)
        for idx, (x, y, z) in self.canonical_points.items():
            # Plot the dot
            ax.plot(x, y, 'o', color='#ff4d4d', markersize=6, zorder=5)
            
            # Logic for label placement
            # ID Label (Bold Red)
            offset_y_id = 0.4
            offset_x_id = -0.3
            if idx in [0, 1, 2, 3]: offset_y_id = -0.6 # Baseline points: ID below
            
            ax.text(x + offset_x_id, y + offset_y_id, str(idx), 
                    color='#ff4d4d', fontsize=11, weight='bold', 
                    ha='center', va='center', zorder=6)
            
            # Coordinate Label (Small White)
            # Offset slightly to the right of the point
            coord_str = f"({x:.2f}, {y:.2f}, {z:.2f})"
            
            # Adjust vertical alignment for coordinates to avoid overlapping lines
            va_align = 'bottom'
            offset_y_coord = -0.5
            # if idx in [2, 3]: # Elbows
            #      va_align = 'top' 
            #      offset_y_coord = -0.8
            if idx in [0, 1, 2, 3]: # Baseline
                offset_y_coord = 0.5 # Below the ID
                va_align = 'top'

            ax.text(x, y + offset_y_coord, coord_str, 
                    color='white', fontsize=6, 
                    ha='center', va=va_align, zorder=6,
                    bbox=dict(facecolor='#1f3b5c', edgecolor='none', alpha=0.6))
            
        # 9. Optional: Plot Trajectory Points
        if add_trajectory and trajectory_points is not None:
            traj_xs = [pt[0] for pt in trajectory_points]
            traj_ys = [pt[1] for pt in trajectory_points]
            ax.plot(traj_xs, traj_ys, 'o-', color='yellow', markersize=4, linewidth=1.5, label='Trajectory', zorder=4)
            ax.legend(loc='upper right', fontsize=8)

        # Axis & Layout
        ax.set_aspect('equal')
        
        # Padding
        ax.set_xlim(-(self.COURT_WIDTH/2 + 1), (self.COURT_WIDTH/2 + 1))
        ax.invert_yaxis()
        ax.invert_xaxis()

        
        ax.set_title(f"Court Model (Origin @ Rim)", color='white', pad=15, fontsize=10)
        
        # Turn off axis
        ax.axis('off') 

        if return_ax:
            # Aspect Ratio para não distorcer
            try:
                ax.set_box_aspect([16, 16]) 
            except:
                pass # Matplotlib antigo pode não ter box_aspect

            return ax
        
        plt.tight_layout()
        plt.show()



class CanonicalCourt3D:
    def __init__(self):
        self.BALL_RADIUS = 0.11  # Raio aproximado da bola de basquete em metros

        self.COURT_WIDTH = 15.0
        self.HALF_COURT_LENGTH = 14.0
        
        # Garrafão
        self.KEY_WIDTH = 4.9
        self.KEY_LENGTH = 5.8
        
        # Alturas (Eixo Z)
        self.RIM_HEIGHT = 3.05
        self.BACKBOARD_BOTTOM = 2.90
        self.BACKBOARD_TOP = 3.95
        
        # Offsets
        self.BACKBOARD_OFFSET = 1.2
        self.RIM_OFFSET = 1.575  # Origem (0,0,0) projetada no chão abaixo do aro
        
        # Raios
        self.RIM_RADIUS = 0.225
        self.THREE_POINT_RADIUS = 6.75
        self.FREE_THROW_RADIUS = 1.8
        self.NO_CHARGE_RADIUS = 1.25
        self.THREE_POINT_SIDE_OFFSET = 0.90

        # Pontos de Interesse (X, Y, Z)
        self.canonical_points = self._define_canonical_points()

    def _define_canonical_points(self):
        """
        Define pontos 3D.
        Origem (0,0,0) = Centro do aro PROJETADO NO CHÃO.
        """
        pts = {}
        y_base = -self.RIM_OFFSET
        y_ft = self.KEY_LENGTH - self.RIM_OFFSET 
        x_key = self.KEY_WIDTH / 2
        x_3pt = (self.COURT_WIDTH / 2) - self.THREE_POINT_SIDE_OFFSET
        z_rim = self.RIM_HEIGHT

        # --- mesmos IDs, mesmos significados ---
        pts[0] = ( -x_3pt, y_base, 0)
        pts[2] = ( x_key, y_base, 0)
        pts[1] = ( -x_key, y_base, 0)
        pts[4] = ( x_key, y_ft, 0)
        pts[3] = ( -x_key, y_ft, 0)

        return pts
    
    def get_canonical_points(self):
        return [self.canonical_points[i] for i in sorted(self.canonical_points.keys())]

    def _get_arc_points(self, center, radius, theta_start, theta_end, z_level=0, n=50):
        """Gera arrays X, Y, Z para desenhar arcos"""
        thetas = np.linspace(np.radians(theta_start), np.radians(theta_end), n)
        xs = center[0] + radius * np.cos(thetas)
        ys = center[1] + radius * np.sin(thetas)
        zs = np.full_like(xs, z_level)
        return xs, ys, zs

    def plot_court(self, add_trajectory: bool = False, 
                   trajectory_points: list = None, 
                   shooter_pos:Tuple[float,float]=None,
                   return_ax:bool=False,
                   ax=None):
        # Configuração da Figura 3D
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        
        plt.rcParams['figure.facecolor'] = '#1f3b5c'
        ax.set_facecolor('#1f3b5c')
        # Remover grid padrão e cor de fundo dos painéis para visual mais limpo
        ax.grid(False)
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        c_line = 'white'
        w_line = 2

        # Variáveis Geométricas
        y_base = - self.RIM_OFFSET
        y_ft = self.KEY_LENGTH - self.RIM_OFFSET
        y_half = self.HALF_COURT_LENGTH - self.RIM_OFFSET

        # ---------------------
        # 1. CHÃO (Z=0)
        # ---------------------
        
        # Limites da Quadra
        x_court = [-self.COURT_WIDTH/2, self.COURT_WIDTH/2, self.COURT_WIDTH/2, -self.COURT_WIDTH/2, -self.COURT_WIDTH/2]
        y_court = [y_base, y_base, y_half, y_half, y_base]
        z_court = [0, 0, 0, 0, 0]
        ax.plot(x_court, y_court, z_court, color=c_line, linewidth=w_line)

        # Garrafão
        x_key = [-self.KEY_WIDTH/2, self.KEY_WIDTH/2, self.KEY_WIDTH/2, -self.KEY_WIDTH/2, -self.KEY_WIDTH/2]
        y_key = [y_base, y_base, y_ft, y_ft, y_base]
        z_key = [0, 0, 0, 0, 0]
        ax.plot(x_key, y_key, z_key, color=c_line, linewidth=w_line)

        # Círculo Lance Livre (Completo)
        fx, fy, fz = self._get_arc_points((0, y_ft), self.FREE_THROW_RADIUS, 0, 360)
        ax.plot(fx, fy, fz, color=c_line, linewidth=w_line)

        # Linha de 3 Pontos
        # Parte Reta
        x_str = (self.COURT_WIDTH/2) - self.THREE_POINT_SIDE_OFFSET
        dy = np.sqrt(self.THREE_POINT_RADIUS**2 - x_str**2)
        ax.plot([-x_str, -x_str], [y_base, dy], [0, 0], color=c_line, linewidth=w_line)
        ax.plot([x_str, x_str], [y_base, dy], [0, 0], color=c_line, linewidth=w_line)
        
        # Parte Curva
        theta_deg = np.degrees(np.arccos(x_str / self.THREE_POINT_RADIUS))
        
        # CORREÇÃO AQUI: mudamos de ax, ay, az para arc_x, arc_y, arc_z
        arc_x, arc_y, arc_z = self._get_arc_points((0,0), self.THREE_POINT_RADIUS, theta_deg, 180-theta_deg)
        ax.plot(arc_x, arc_y, arc_z, color=c_line, linewidth=w_line)

        # No Charge Semi-Circle
        nx, ny, nz = self._get_arc_points((0,0), self.NO_CHARGE_RADIUS, 0, 180)
        ax.plot(nx, ny, nz, color=c_line, linewidth=w_line)

        # ---------------------
        # 2. ESTRUTURA VERTICAL (Z > 0)
        # ---------------------
        
        y_bb = y_base + self.BACKBOARD_OFFSET # Y da tabela

        # Poste (simplificado)
        # Base até base da tabela
        ax.plot([0, 0], [y_base-0.5, y_base-0.5], [0, 3.0], color='gray', linewidth=4)
        # Suporte vertical da tabela
        ax.plot([0, 0], [y_base-0.5, y_bb], [3.0, 3.0], color='gray', linewidth=4)

        # Tabela (Backboard)
        bx = [-0.9, 0.9, 0.9, -0.9, -0.9]
        by = [y_bb, y_bb, y_bb, y_bb, y_bb]
        bz = [self.BACKBOARD_BOTTOM, self.BACKBOARD_BOTTOM, self.BACKBOARD_TOP, self.BACKBOARD_TOP, self.BACKBOARD_BOTTOM]
        ax.plot(bx, by, bz, color='white', linewidth=2)

        # Aro (Rim) - Z = 3.05
        rx, ry, rz = self._get_arc_points((0,0), self.RIM_RADIUS, 0, 360, z_level=self.RIM_HEIGHT)
        ax.plot(rx, ry, rz, color='orange', linewidth=3)
        
        # Conexão Tabela-Aro
        ax.plot([0, 0], [y_bb, 0], [self.RIM_HEIGHT, self.RIM_HEIGHT], color='orange', linewidth=3)

        # ---------------------
        # 3. PONTOS CANÔNICOS
        # ---------------------
        for idx, (x, y, z) in self.canonical_points.items():
            color = 'red' if idx == 12 else '#ff4d4d'
            ax.scatter(x, y, z, color=color, s=50, depthshade=False)
            
            # Label
            ax.text(x, y, z + 0.3, str(idx), color='white', fontsize=10, weight='bold', ha='center')

        
        # Optional trajectory points
        if add_trajectory and trajectory_points is not None:
            traj_xs = [pt[0] for pt in trajectory_points]
            traj_ys = [pt[1] for pt in trajectory_points]
            traj_zs = [pt[2] for pt in trajectory_points]
            ax.plot(traj_xs, traj_ys, traj_zs, 'o-', color='yellow', markersize=4, linewidth=1.5, label='Trajectory', zorder=4)
            ax.legend(loc='upper right', fontsize=8)

        if shooter_pos is not None:
            ax.scatter(x, y, color='black', s=50)
            ax.text(x, y, 0.5, "Shooter Position", color='white', fontsize=7, weight='bold', ha='center')

        # ---------------------
        # 4. CONFIGURAÇÃO FINAL
        # ---------------------
        
        ax.set_title("3D Canonical Court", color='white', pad=20)
        ax.set_xlabel("X (Width)", color='white')
        ax.set_ylabel("Y (Length)", color='white')
        ax.set_zlabel("Z (Height)", color='white')
        ax.tick_params(colors='white')

        # Limites
        ax.set_xlim(-(self.COURT_WIDTH/2 + 1), (self.COURT_WIDTH/2 + 1))
        ax.set_ylim(y_base - 2, y_half)
        ax.set_zlim(0, 5)

        ax.invert_yaxis()
        ax.invert_xaxis()

        # Aspect Ratio
        ax.set_box_aspect([self.COURT_WIDTH, self.HALF_COURT_LENGTH + 2, 5]) 

        # Visão Inicial
        ax.view_init(elev=30, azim=-45)

        if return_ax:
                # Aspect Ratio para não distorcer
            try:
                ax.set_box_aspect([16, 16, 8]) 
            except:
                pass # Matplotlib antigo pode não ter box_aspect

            return ax

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":

    court2d = CanonicalCourt2D()
    court3d = CanonicalCourt3D()

    for i in range(4):
        p2 = court2d.canonical_points[i]
        p3 = court3d.canonical_points[i]

        assert np.allclose(p2[:2], p3[:2]), f"Mismatch at point {i}"
        assert p3[2] == 0

    print("Pontos 3D (X, Y, Z):")
    for k, v in court3d.canonical_points.items():
        print(f"ID {k}: {v}")
        
    court2d.plot_court()
    court3d.plot_court()

