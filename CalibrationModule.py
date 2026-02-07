# generic imports
from typing import List, Tuple, Optional, Literal
from dataclasses import dataclass
import os
import itertools
import json 

# task specific imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# project imports
from CourtModule import CanonicalCourt2D, CanonicalCourt3D


# Path
CALIBRATION_IMG = "C:/Documents/Polimi/IACV/ProjectAssignment/Project/data/calibration/court_frontal.png"
assert os.path.exists(CALIBRATION_IMG)
METADATA = "C:/Documents/Polimi/IACV/ProjectAssignment/Project/data/calibration/metadata.json"
assert os.path.exists(METADATA)

# Data Classes

@dataclass
class VanishingPoint:
    point_2d: Tuple[float, float]
    direction_3d: Optional[Literal["x", "y", "z"]] = None

@dataclass
class ParallelLines:
    lines: List[Tuple[Tuple[int, int], Tuple[int, int]]]
    vanishing_point: Optional[VanishingPoint] = None
    real_distance_btw = Optional[float]
    direction_3d: Optional[Literal["x", "y", "z"]] = None


# Camera Calibration Classes


class CameraModel:
    """
    Docstring for CameraModel

    Parameters:
    K: (3x3) intrinsics -> type: np.ndarray
    R: (3x3) rotation matrix (world → camera) -> type: np.ndarray
    t: (3x1) translation vector (world → camera) -> type: np.ndarray
    dist_coeffs: Optional (kx1) distortion coefficients -> type: np.ndarray
    """
    def __init__(self, K, R, t, dist_coeffs: Optional[np.ndarray] = None): # Distortion coefficients are rarely not 0 and can cause overfitting
        self.K = K.astype(np.float64)
        self.R = R.astype(np.float64)
        self.t = t.reshape(3, 1).astype(np.float64)

        self.focal_length = (self.K[0, 0] + self.K[1, 1]) / 2

        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros((4,1))

        # Projection matrix
        self.P = self.K @ np.hstack((self.R, self.t))

        # Cache inverses
        self.K_inv = np.linalg.inv(self.K)
        self.R_inv = self.R.T

        # Camera center in world coordinates
        self.C = -self.R_inv @ self.t
        
        self.video_fps = (None, None) # To be set externally if needed

    def get_fps(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        self.video_fps = (video_path, fps)
        return fps
        

    def world_to_camera(self, Xw):
        """
        Xw: (3,) or (N,3)
        Returns: Xc in camera coordinates
        """
        Xw = np.atleast_2d(Xw).T  # (3,N)
        Xc = self.R @ Xw + self.t
        return Xc.T
    
    def camera_to_world(self, Xc):
        """
        Xc: (3,) or (N,3)
        """
        Xc = np.atleast_2d(Xc).T
        Xw = self.R_inv @ (Xc - self.t)
        return Xw.T

    def project_world(self, Xw):
        """
        Projects world points to image.
        Xw: (N,3)
        Returns: (N,2) pixel coordinates
        """
        Xw = np.asarray(Xw, dtype=np.float64)

        rvec, _ = cv2.Rodrigues(self.R)
        img_pts, _ = cv2.projectPoints(
            Xw,
            rvec,
            self.t,
            self.K,
            self.dist_coeffs
        )
        return img_pts.reshape(-1, 2)
    
    def backproject_ray(self, u, v):
        """
        Given pixel (u,v), returns:
        - ray origin (camera center in world coords)
        - ray direction (unit vector in world coords)
        """
        pixel = np.array([[u, v, 1.0]]).T
        ray_cam = self.K_inv @ pixel
        ray_cam /= np.linalg.norm(ray_cam)

        # Convert ray to world coordinates
        ray_world = self.R_inv @ ray_cam
        ray_world /= np.linalg.norm(ray_world)

        origin = self.C.flatten()
        direction = ray_world.flatten()

        return origin, direction
    
    def depth_of_world_point(self, Xw):
        """
        Returns Zc (depth) in camera coordinates
        """
        Xc = self.world_to_camera(Xw)
        return Xc[:, 2]

    def is_in_front(self, Xw):
        """
        Check if world points are in front of the camera
        """
        return self.depth_of_world_point(Xw) > 0
    
    def project_pixel_to_ground(self, u, v):
        """
        Ray Casting: Projects one pixel (u,v) to plane Z=0 (ground).
        Returns (x_ground, y_ground).
        """
        
        uv_hom = np.array([u, v, 1.0], dtype=np.float64).reshape(3, 1)
        ray_camera = self.K_inv @ uv_hom

       
        ray_world = self.R.T @ ray_camera
        ray_world = ray_world.flatten()
      
        C = -self.R.T @ self.t
        C = C.flatten()
        
        if abs(ray_world[2]) < 1e-6: 
            return None

        s = -C[2] / ray_world[2]
        
        # Ponto de interseção
        P_ground = C + s * ray_world
        
        return P_ground[0], P_ground[1]

    def summary(self):
        print("Camera Model Summary")
        print("--------------------")
        print("Intrinsics K:\n", self.K)
        print("\nRotation R:\n", self.R)
        print("\nTranslation t:\n", self.t.flatten())
        print("\nCamera center C (world):\n", self.C.flatten())

class CalibrationDebugger:
    def __init__(self, camera: CameraModel, court_2d: CanonicalCourt2D, image_path: str):
        self.camera = camera
        self.court_2d = court_2d

        # --- Check path ---
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # --- Load image with OpenCV ---
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"cv2.imread failed to load image: {image_path}")

        # --- Convert to RGB for matplotlib ---
        self.image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        self.image_path = image_path
        self.image_shape = self.image.shape
    
    def check_in_court_projection(self, object_pts_3d):
        img_pts = self.camera.project_world(object_pts_3d)

        plt.figure(figsize=(8, 5))
        plt.imshow(self.image)
        plt.scatter(img_pts[:, 0], img_pts[:, 1], c='r', s=30)
        plt.title("Projection of court points onto image")
        plt.axis("off")
        plt.show()
    
    def check_ground_point_depth(self, object_pts_3d):
        depths = self.camera.depth_of_world_point(object_pts_3d)

        for pt, depth in zip(object_pts_3d, depths):
            print(f"Point {pt} has depth {depth:.4f}")
            if depth < 0: 
                print(f"The depth on the Point {pt} indicates error")

        for depth in depths:
            assert depth > 0, (
                "Some ground points are behind the camera! "
                "All depths must satisfy Z > 0."
            )
    
    def check_backprojection_on_ground(self, u, v):
        """
        Checks whether the backprojected ray from pixel (u, v)
        correctly intersects the ground plane Z = 0.
        """
        origin, direction = self.camera.backproject_ray(u, v)

        if abs(direction[2]) < 1e-9:
            raise ValueError("Backprojection ray is parallel to ground plane!")

        t = -origin[2] / direction[2]
        X_ground = origin + t * direction

        if abs(X_ground[2]) > 1e-3:
            print("Backprojected point is NOT on the ground plane!")
            print(f"Z = {X_ground[2]:.6f}")
        else:
            print(f"Backprojected ground point: {X_ground}")



# Display Calibration Classes


class CalibrationDisplay:
    MAX_W, MAX_H = 1400, 900
    def compute_display_scale(self, w, h, max_w=MAX_W, max_h=MAX_H):
        return min(max_w / w, max_h / h, 1.0)

    def __init__(self, image_path: str = CALIBRATION_IMG):
        self.image_path = image_path
        raw_img = cv2.imread(self.image_path)
        h, w = raw_img.shape[:2]
        self.h_raw = h
        self.w_raw = w
        self.scale = self.compute_display_scale(w, h)
        self.img = cv2.resize(raw_img, (int(w * self.scale), int(h * self.scale)), interpolation=cv2.INTER_AREA)
        self.canvas = self.img.copy()

    def clear_canvas(self):
        self.canvas = self.img.copy()
    
    def save_canvas(self, save_path: str):
        cv2.imwrite(save_path, self.canvas)

    def draw_lines(self):
        window = "Draw Lines"
        lines = []
        start = None
        drawing = False
        img_preview = self.canvas.copy()

        def event_handler(event, x, y, flags, param):
            nonlocal start, drawing, img_preview
            if event == cv2.EVENT_LBUTTONDOWN:
                start = (x, y)
                drawing = True
            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    img_preview = self.canvas.copy()
                    cv2.line(img_preview, start, (x, y), (0, 255, 0), 2)
                    cv2.imshow(window, img_preview)
            elif event == cv2.EVENT_LBUTTONUP:
                end = (x, y)
                drawing = False
                lines.append((start, end))
                cv2.line(self.canvas, start, end, (0, 255, 0), 2)
                cv2.imshow(window, self.canvas)
                print(f"Start: {start}, End: {end}")

        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, event_handler)

        while True:
            cv2.imshow(window, img_preview)
            key = cv2.waitKey(1) & 0xFF 
            if key == ord('u'):
                if lines:
                    lines.pop()
                    self.clear_canvas()
                    for start, end in lines:
                        cv2.line(self.canvas, start, end, (0, 255, 0), 2)
                    img_preview = self.canvas.copy()
            elif key == ord('q'):
                break
        cv2.destroyAllWindows()
        return lines
    
    def draw_points(self):
        window = "Draw Points"
        points = []

        def event_handler(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                cv2.circle(self.canvas, (x, y), 4, (0, 0, 255), -1)
                print(f"Point: {(x, y)}")

        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, event_handler)

        while True:
            cv2.imshow(window, self.canvas)
            key = cv2.waitKey(1) & 0xFF 
            if key== ord('q'): # Press 'q' to exit
                break
            elif key == ord('u'):
                if points:
                    points.pop()
                    self.clear_canvas()
                    for point in points:
                        cv2.circle(self.canvas, point, 4, (0, 0, 255), -1)
                    cv2.imshow(window, self.canvas)
        cv2.destroyAllWindows()

        print("Resizing to original pixel dimensions...")
        original_scale_points = []
        for (px, py) in points:
            orig_x = px / self.scale
            orig_y = py / self.scale
            original_scale_points.append((orig_x, orig_y))

        return original_scale_points


# Functions

def find_line_intersection(l1: Tuple[Tuple[int, int], Tuple[int, int]], 
                       l2: Tuple[Tuple[int, int], Tuple[int, int]]) -> Optional[Tuple[float, float]]:
    x1, y1 = l1[0]
    x2, y2 = l1[1]
    x3, y3 = l2[0]
    x4, y4 = l2[1]

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        raise ValueError("Lines do not intersect")

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / denom

    return np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)])


def find_vanishing_point(lines: ParallelLines) -> VanishingPoint:
    approximate_vanishing_points = []
    for line1, line2 in itertools.combinations(lines.lines, 2):
        try:
            intersection = find_line_intersection(line1, line2)
        except ValueError:
            continue
        approximate_vanishing_points.append(intersection)
    
    vp = VanishingPoint(point_2d=np.mean(np.array(approximate_vanishing_points), axis=0), direction_3d=lines.direction_3d)
    lines.vanishing_point = vp
    return vp
        
def define_calibration_points(calibration_display:CalibrationDisplay, court3D:CanonicalCourt2D=CanonicalCourt3D()):
    court = court3D
    court.plot_court()
    print("Press 'q' to quit drawing mode and 'u' to undo last action.")
    img_points_2d = calibration_display.draw_points()
    real_points_3d = [v for _, v in sorted(court.canonical_points.items())]
    return img_points_2d, real_points_3d

def get_calibration_points(calibration_display: Optional[CalibrationDisplay], 
                           recover_from_file: Optional[bool]=False,
                           metadata_file: Optional[bool]=None,
                           save_to_file: bool=False,
                           court2D: CanonicalCourt2D=CanonicalCourt2D()) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float, float]]]:
    assert os.path.exists(metadata_file)
    if recover_from_file:
        with open (metadata_file, 'r') as f:
            metadata = json.load(f)
            return metadata["calibration"]["points_2d"], metadata["calibration"]["points_3d"]
    else:
        metadata = {"calibration": {"points_2d": [],"points_3d": []}}
        img_2d, real_3d = define_calibration_points(calibration_display, court2D)
        if save_to_file:
            with open (metadata_file, 'r+') as f:
                metadata = json.load(f)
                metadata["calibration"]["points_2d"] = img_2d
                metadata["calibration"]["points_3d"] = real_3d
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=8)
        return img_2d, real_3d

def get_z_plane_homography(img_points: List[Tuple[float, float]], real_points: List[Tuple[float, float, float]], 
                     z_plane: float = 0.0, save_to_file: bool=False, metadata_file: Optional[str]=None) -> np.ndarray:
    
    assert len(img_points) == len(real_points) and len(img_points) >= 4, "At least 4 points are required for homography computation."
    
    src_pts = np.array(img_points, dtype=np.float32)
    dst_pts = np.array([[x, y] for x, y, z in real_points if abs(z - z_plane) < 1e-5], dtype=np.float32)

    H, inliers = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    if save_to_file and metadata_file is not None:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        metadata["HomographyZ"] = H.tolist() if H is not None else None
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=8)
    return H, inliers



def refine_focus_optimization(real_pts, img_pts, f_init, rvec, tvec, h, w):

    def reprojection_error(params, real_pts, img_pts, h, w): # Local Bundle Adjustment
        f, rx, ry, rz, tx, ty, tz = params

        K = np.array([
            [f, 0, w / 2],
            [0, f, h / 2],
            [0, 0, 1]
        ])

        rvec = np.array([[rx], [ry], [rz]])
        tvec = np.array([[tx], [ty], [tz]])

        projected, _ = cv2.projectPoints(
            real_pts, rvec, tvec, K, None
        )

        projected = projected.squeeze()
        return (projected - img_pts).ravel()


    x0 = np.array([
        f_init,
        rvec[0,0], rvec[1,0], rvec[2,0],
        tvec[0,0], tvec[1,0], tvec[2,0]
    ])

    res = least_squares(
    reprojection_error,
    x0,
    method='lm',
    args=(real_pts, img_pts, h, w)
    )

    return res



def calibrate_camera_pnp(img_pts, real_pts, h, w, metadata_file=None, camera_height_threshold: Tuple[float, float]=(0.5, 5.0)):
    '''
    Calculates Intrinsic and Extrinsic Parameters of Camera
    '''
    # Assume image center as starting principal
    cx, cy = w / 2, h / 2
    # Initial focal length estimate
    f_initial = 1.2*w

    K = np.array([   #Camera Matrix
        [f_initial, 0, cx],
        [0, f_initial, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    dist_coeffs = np.zeros((4, 1)) # Assuming zero distortion intitially

    real_pts = np.array(real_pts, dtype=np.float32)
    img_pts = np.array(img_pts, dtype=np.float32)

    # Solver PnP for coplanar points on the court 
    success, rvec, tvec, = cv2.solvePnP(
        real_pts, 
        img_pts, 
        K, 
        dist_coeffs, 
        flags=cv2.SOLVEPNP_IPPE
    )

    if not success:
        raise "PnP solver failed"

    # Result Extraction
    # Rotatate to 3x3 matrix
    R, _ = cv2.Rodrigues(rvec) # Rotation Matrix

    res = refine_focus_optimization(real_pts=real_pts,
                                    img_pts=img_pts,
                                    f_init=f_initial,
                                    rvec=rvec, tvec=tvec,
                                    h=h, w=w)
    
    f_opt = res.x[0]

    K_final = np.array([
    [f_opt, 0, cx],
    [0, f_opt, cy],
    [0, 0, 1]
    ])

    rvec_final = res.x[1:4].reshape(3,1)
    tvec_final = res.x[4:7].reshape(3,1)

    R_final, _ = cv2.Rodrigues(rvec_final)

    C_final = -R_final.T @ tvec_final

    real_camera_height = C_final[2][0]

    print("tvec =", tvec_final)
    print("Camera center (world calib) =", C_final)
    print("R[2,:] (camera Z axis in world) =", R_final.T[:,2])

    height_verification = real_camera_height
    print(f"Refined Focal Length: {f_opt}")
    print(f"Camera Height above court: {height_verification} units (Meters -> court units are in meters)")
    if height_verification > camera_height_threshold[1] or height_verification < camera_height_threshold[0]:
        print("Warning('Unusual camera height detected. Please verify calibration points.')")


    calibration_results = {
        "intrinsic_matrix": K_final.tolist(),
        "rotation_matrix": R_final.tolist(),
        "translation_vector": tvec_final.tolist(),
        "focal_length": float(K_final[0, 0]),
        "camera_height": float(real_camera_height) # height in relation to the court  
    }


    if metadata_file:
        with open(metadata_file, 'r+') as f:
            data = json.load(f)
            data["camera_params"]["calibration_pnp"] = calibration_results
            f.seek(0)
            json.dump(data, f, indent=4)

    return calibration_results