from typing import Optional, List, Tuple, Dict
import numpy as np  
import cv2
import json
import os
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from CalibrationModule import *
from TrackingModule import *


from modules.CalibrationModule import CameraModel
from modules.CourtModule import CanonicalCourt3D

class TrajectoryEstimator:
    def __init__(self, court_model: CanonicalCourt3D, camera_model: CameraModel, manual_fps: Optional[float] = None):
        self.court = court_model
        self.cam = camera_model
        if manual_fps is not None:
            self.fps = manual_fps
        else:
            if self.cam.video_fps[1] is not None:
                self.fps = self.cam.video_fps[1] 
            else:
                raise ValueError("FPS not available from camera model. Please provide manual_fps.")
        self.g = 9.81  # Gravidade (m/s^2)

    def _predict_3d_point(self, params, t):
        x0, y0, z0, vx, vy, vz = params
        px = x0 + vx * t
        py = y0 + vy * t
        pz = z0 + vz * t - 0.5 * self.g * (t**2)
        return np.array([px, py, pz])
    
    def estimate_depth_from_radius(self, r_px): # even though this method is not well equipped to deal with true trajectory estimation 
                                      # due to bluring and instability, it serves the purpose of limiting and helping depth positioning
        true_r = self.court.BALL_RADIUS
        focal_length = self.cam.focal_length
        if r_px > 0:
            # Formula da projeção perspectiva
            return (focal_length * true_r) / r_px
        else:
            return None 
        

    def residual_function(self, params, tracking_data, constraints: Dict[str, float] = None):
        """
        Parameters:
        params (List[float]): [x0, y0, z0, vx, vy, vz]
        tracking_data (List[Dict]): List of tracking points with 'x', 'y', 'frame', 'r' keys.
        constraints (Dict[str, float], optional): Physical constraints:
        E.g., {'start_z': value, 'end_z': value, 'end_y': value, 'start_y': value}
        """
        # Weights # Peso alto para forçar a restrição (Hard Constraint via Penalty)
        depth_constraint = 20.0 
        base_constraint = 10.0
        anchor_constraint = 100.0
        regularization_constraint = 0.2
        strong_weight = 10000.0
        residuals = []
        start_frame = tracking_data[0]['frame']

        for point in tracking_data:
            t = (point['frame'] - start_frame) / self.fps
            
            P_world = self._predict_3d_point(params, t)
            
            proj_uv = self.cam.project_world(P_world.reshape(1, 3)).flatten()

            proj_w = self.cam.depth_of_world_point(P_world)
            proj_w = proj_w.item()
            estimated_depth = self.estimate_depth_from_radius(r_px=point['r'])

          
            residuals.append((proj_uv[0] - point['x'])*base_constraint)
            residuals.append((proj_uv[1] - point['y'])*base_constraint)
            if estimated_depth is not None:
                residuals.append((proj_w-estimated_depth)*depth_constraint) 
            else:
                print(f"Estimated Depth at frame {point['frame']} is None")

        if constraints:

            if 'start_x' in constraints:
                err = (params[0] - constraints['start_x']) * anchor_constraint
                residuals.append(err)
            
            if 'start_z' in constraints:
                z_start = params[2] # z0
                err = (z_start - constraints['start_z']) * anchor_constraint
                residuals.append(err)

            if 'start_y' in constraints:
                y_start = params[1]
                err1 = (y_start - constraints['start_y']) * anchor_constraint

                y_bb = -self.court.RIM_OFFSET + self.court.BACKBOARD_OFFSET + self.court.BALL_RADIUS
                penetration = max(0, y_bb - y_start)
                err2 = penetration * strong_weight

                err = np.sqrt(max(err1**2, err2**2))
                residuals.append(err)
                ("start_y in constraints has added residual of size ", err)

            if 'end_z' in constraints:
                last_t = (tracking_data[-1]['frame'] - start_frame) / self.fps
                z_end_pred = params[2] + params[5] * last_t - 0.5 * self.g * (last_t**2)
                err = (z_end_pred - constraints['end_z']) * anchor_constraint
                residuals.append(err)

            if 'end_y' in constraints:
                last_t = (tracking_data[-1]['frame'] - start_frame) / self.fps
                y_end_pred = params[1] + params[4] * last_t
                err1 = (y_end_pred - constraints['end_y']) * anchor_constraint

                y_bb = -self.court.RIM_OFFSET + self.court.BACKBOARD_OFFSET + self.court.BALL_RADIUS
                penetration = max(0, y_bb - y_end_pred)
                err2 = penetration * strong_weight

                err = np.sqrt(max(err1**2, err2**2))
                residuals.append(err)
                
            
        vx, vy, vz = params[3], params[4], params[5]
        residuals.append(vx * regularization_constraint) 
        residuals.append(vy * regularization_constraint) 
        residuals.append(vz * regularization_constraint) 

        for point in tracking_data:
            t = (point['frame'] - start_frame) / self.fps

            y_t = params[1]
            m1 = np.maximum(0, y_t-self.court.HALF_COURT_LENGTH) 
            m2 = np.maximum (m1, -y_t)
            err_depth = m2 * depth_constraint 
            
            z_t = params[2] + params[5] * t - 0.5 * self.g * (t**2)
            err_floor = np.maximum(0, -z_t) * base_constraint 
            
            residuals.append(err_floor)
            residuals.append(err_depth)

        return np.array(residuals)

    def detect_bounce_split_index(self, tracking_data):
        """
        Docstring for detect_bounce_split_index
        Detects a bounce in the trajectory by analyzing vertical motion.
        Returns the index in tracking_data where the bounce occurs, or None if not found.

        Parameters:
        tracking_data (List[Dict]): List of tracking points with 'y' and 'frame' keys.
        """
        if len(tracking_data) < 5: return None

        ys = np.array([p['y'] for p in tracking_data])
        
        kernel = np.array([0.25, 0.5, 0.25])
        ys_smooth = np.convolve(ys, kernel, mode='valid')

        dys = np.diff(ys_smooth)
        
    
        
        for i in range(len(dys) - 1):
            if dys[i] > 1.0 and dys[i+1] < -1.0:
                 return i + 2
        return None

    def fit_segment(self, segment_data, specific_constraints: Dict = None):
        """
        Docstring for fit_segment
        Fits a single trajectory segment using optimization.

        Parameters:
        segment_data (List[Dict]): Tracking data for the segment.
        specific_constraints (Dict, optional): Constraints for this segment: 
        E.g., {'start_z': value, 'end_z': value, 'end_y': value, 'start_y': value}.
        """
        if len(segment_data) < 6: # Mínimo para 6 variaveis
            return None
        
        # defining bounds
        lower_bounds = [-10, -1.5,  0.0, -15, -15, -15]
        upper_bounds = [ 10, 20, 10.0,  15,  15,  15]
        if specific_constraints:
            if 'start_z' in specific_constraints:
                sz = specific_constraints['start_z']
                lower_bounds[2] = sz - 0.5
                upper_bounds[2] = sz + 0.5
            if 'start_y' in specific_constraints:
                sy = specific_constraints['start_y']
                lower_bounds[1] = sy - 0.5
                upper_bounds[1] = sy + 0.5
            if 'start_x' in specific_constraints:
                sx = specific_constraints['start_x']
                lower_bounds[0] = sx - 0.5
                upper_bounds[0] = sx + 0.5

        r0 = segment_data[0]['r']
        initial_depth_guess = 5.0 # Default 
        if r0 > 0:
            # Z_cam = f * R / r
            z_cam = (self.cam.focal_length * self.court.BALL_RADIUS) / r0
            
            initial_depth_guess = z_cam

        # [x0, y0, z0, vx, vy, vz]
        x0_guess = [0, initial_depth_guess, 2.0, 0, 5, 5] 
        
        linear_guess = self.solve_linear_system(segment_data)
        if linear_guess:
            valid = True
            lg = [linear_guess['x0'], linear_guess['y0'], linear_guess['z0']]
            for i in range(3):
                if lg[i] < lower_bounds[i] or lg[i] > upper_bounds[i]: valid = False
            
            if valid:
                x0_guess = [
                    linear_guess['x0'], linear_guess['y0'], linear_guess['z0'],
                    linear_guess['vx'], linear_guess['vy'], linear_guess['vz']
                ]
            
        if specific_constraints:
            if 'start_z' in specific_constraints: x0_guess[2] = specific_constraints['start_z']
            if 'start_y' in specific_constraints: x0_guess[1] = specific_constraints['start_y']
            if 'start_x' in specific_constraints: x0_guess[0] = specific_constraints['start_x']
        try:
            res = least_squares(
                self.residual_function, 
                x0=x0_guess, 
                bounds=(lower_bounds, upper_bounds),
                args=(segment_data, specific_constraints), 
                loss='soft_l1', 
                f_scale=10.0    
            )
            
            opt = res.x
            return {
                "start_frame": segment_data[0]['frame'],
                "end_frame": segment_data[-1]['frame'],
                "x0": opt[0], "y0": opt[1], "z0": opt[2],
                "vx": opt[3], "vy": opt[4], "vz": opt[5]
            }
        except Exception as e:
            print(f"Otimização falhou: {e}")
            return None

    
    def _is_rim_bounce(self, bounce_pixel_uv, threshold_pixels=300):
        """
        Docstring for _is_rim_bounce
        Determines if the bounce is close enough to the rim in pixel space.

        Parameters:
        bounce_pixel_uv (np.array): Pixel coordinates of the bounce [u, v].
        threshold_pixels (float): Distance threshold in pixels to consider as rim bounce.

        Returns:
        bool: True if bounce is within threshold of rim, False otherwise.
        """
        rim_center_3d = np.array([[0.0, 0.0, self.court.RIM_HEIGHT]]) # [X, Y, Z]
        
        rim_pixel = self.cam.project_world(rim_center_3d).flatten() # Return [u, v]
        
        distu = rim_pixel[0] - bounce_pixel_uv[0]
        distv = rim_pixel[1] - bounce_pixel_uv[1]
        dist = np.sqrt(distv**2 + distu**2)
        
        print(f"Distância do quique ao aro: {dist:.2f} pixels")
        return dist < threshold_pixels

   
    def multi_segment_reconstruct(self, tracking_data, shooter_ground_pos: Tuple[int,int], proximity_threshold=300):
        """
        Docstring for multi_segment_reconstruct
        Reconstructs trajectory possibly with multiple segments (e.g., shot + bounce).

        Parameters:
        tracking_data (List[Dict]): Full tracking data for the trajectory.
        shooter_ground_pos: Tuple (X, Y) of position of player's feet in 3D.

        Returns:
        List of segments with their fitted parameters.
        """

        feet_constraints = {}
        if shooter_ground_pos:
            print(f">> Usando âncora dos pés: X={shooter_ground_pos[0]:.2f}, Y={shooter_ground_pos[1]:.2f}")
            feet_constraints['start_x'] = shooter_ground_pos[0]
            feet_constraints['start_y'] = shooter_ground_pos[1]

        split_idx = self.detect_bounce_split_index(tracking_data)
        results = []
        
        if split_idx:
            bounce_frame_data = tracking_data[split_idx]
            bounce_pixel = np.array([bounce_frame_data['x'], bounce_frame_data['y']])
            
            print(f"Bounce detectado no frame {bounce_frame_data['frame']}.")
            
            if self._is_rim_bounce(bounce_pixel, threshold_pixels=proximity_threshold):
                print(f">> Bounce Type: Rim (Z={self.court.RIM_HEIGHT})")
                bounce_height = self.court.RIM_HEIGHT 
                bounce_depth = self.court.BALL_RADIUS*0.5
                shot_specific_constraints={'end_z': bounce_height, 
                                           'end_y':bounce_depth,
                                           **feet_constraints}
                bounce_specific_constraints={'start_z': bounce_height, 
                                             'start_y':bounce_depth
                                             }

            else:
                print(">> Bounce Type: Ground (Z=0.0)")
                bounce_height = self.court.BALL_RADIUS 
                shot_specific_constraints={'end_z': bounce_height,
                                           **feet_constraints}
                bounce_specific_constraints={'start_z': bounce_height}
                
            
            shot_data = tracking_data[:split_idx+1]
            shot_params = self.fit_segment(shot_data, specific_constraints=shot_specific_constraints) 
            
            if shot_params:
                results.append({"type": "shot", "params": shot_params})
            
            bounce_data = tracking_data[split_idx:]
            # A restrição inicial deve ser a mesma altura (continuidade)
            bounce_params = self.fit_segment(bounce_data, specific_constraints=bounce_specific_constraints)
            
            if bounce_params:
                results.append({"type": "bounce", "params": bounce_params})
                
        else:
            print("Unique trajectory segment detected.")
            params = self.fit_segment(tracking_data, specific_constraints=feet_constraints)
            if params:
                results.append({"type": "shot", "params": params})
                
        return results

    def solve_linear_system(self, tracking_data):
        """
        Docstring for solve_linear_system
        Solves a linear system to get an initial guess for trajectory parameters.

        Parameters:
        tracking_data (List[Dict]): List of tracking points with 'x', 'y', 'frame', 'r' keys.
        """
        if len(tracking_data) < 6: return None
        
        P = self.cam.P
        start_frame = tracking_data[0]['frame']
        
        A = []
        b = []

        for point in tracking_data:
            u, v = point['x'], point['y']
            t = (point['frame'] - start_frame) / self.fps
            
            
            
            #X = X0 + Vx*t
            # L1: X0, Y0, Z0, Vx, Vy, Vz
            
            p1 = P[0, :] - u * P[2, :]
            p2 = P[1, :] - v * P[2, :]
            
            # Coeficients [X0, Y0, Z0, Vx, Vy, Vz]
            # X(t) = X0 + Vx*t
            
            # Linha u: p1 . [X, Y, Z, 1]^T = 0
            # p1_0*(X0+Vx*t) + p1_1*(Y0+Vy*t) + p1_2*(Z0+Vz*t - 0.5gt^2) + p1_3 = 0
            
            grav_term = -0.5 * self.g * (t**2)
            
            row_u = [
                p1[0], p1[1], p1[2],       # Coeffs X0, Y0, Z0
                p1[0]*t, p1[1]*t, p1[2]*t  # Coeffs Vx, Vy, Vz
            ]
            val_u = -p1[3] - p1[2]*grav_term
            
            row_v = [
                p2[0], p2[1], p2[2],
                p2[0]*t, p2[1]*t, p2[2]*t
            ]
            val_v = -p2[3] - p2[2]*grav_term
            
            A.append(row_u)
            A.append(row_v)
            b.append(val_u)
            b.append(val_v)

        A = np.array(A)
        b = np.array(b)
        
        try:
            X, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            return {
                'x0': X[0], 'y0': X[1], 'z0': X[2],
                'vx': X[3], 'vy': X[4], 'vz': X[5]
            }
        except:
            return None

    def generate_3d_points_from_result(self, result_entry, step_t=0.033, output_json: Optional[str] = None):
        """
        Parameters:
        result_entry (Dict): Fitted trajectory parameters.
        step_t (float): Time step in seconds for point generation.
        """
        params = result_entry['params']
        
        total_time = (params['end_frame'] - params['start_frame']) / self.fps
        times = np.arange(0, total_time, step_t)
        
        points = []
        x_flat = [params['x0'], params['y0'], params['z0'], params['vx'], params['vy'], params['vz']]
        
        for t in times:
            pt = self._predict_3d_point(x_flat, t)
            points.append(pt)
        
        if output_json:
            if os.path.exists(output_json):
                with open(output_json, 'r+') as f:
                    existing_data = json.load(f)
                    existing_data["points_3d"].extend([pt.tolist() for pt in points])
                    f.seek(0)  
                    json.dump(existing_data, f, indent=4)
            else:
                with open(output_json, 'w') as f:
                    json.dump({"points_3d": [pt.tolist() for pt in points]}, f, indent=4)
            
        return points

