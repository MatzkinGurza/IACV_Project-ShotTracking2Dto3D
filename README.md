# Monocular Frontal View 2D to 3D Reconstruction of Basketball Trajectory

## Overview

This project implements a cost-effective system for reconstructing the **3D trajectory of a basketball** from a single, static, frontal-perspective camera. By combining geometric camera calibration, automated ball tracking, and physics-informed non-linear optimization, the system transforms noisy 2D video data into accurate 3D metric coordinates.

The solution addresses the **monocular depth ambiguity** problem inherent in frontal views by leveraging semantic scene anchors (such as the shooter's position and the rim height) and physical constraints (gravity and projectile motion).

### Key Features

* **Camera Calibration:** Solves the PnP (Perspective-n-Point) problem using FIBA-standard court landmarks to recover camera intrinsics and extrinsics.
* **Automated Tracking:** Detects and tracks the basketball using Hough Circle Transform and motion heuristics, with resolution scaling for high-fidelity coordinate mapping.
* **Semantic Anchors:** Utilizes **YOLOv8-Pose** to detect the shooter's feet (Start Anchor) and identifies rim bounces (End Anchor) to lock the trajectory in 3D space.
* **Physics Engine:** Fits a parabolic projectile model () to 2D observations using non-linear least squares optimization (Trust Region Reflective algorithm).
* **Visualization:** Generates synchronized side-by-side videos showing the original footage alongside real-time 3D and 2D top-down reconstructions.

---

## Project Structure

The codebase is modularized into four core components:

* **`CourtModule.py`**: Defines the `CanonicalCourt3D` and `CanonicalCourt2D` classes based on official FIBA dimensions. Provides the ground truth 3D world coordinate system.
* **`CalibrationModule.py`**: Handles camera calibration. Implements the `CameraModel` class to manage Intrinsic () and Extrinsic () matrices and provides functions for projecting points between 2D and 3D.
* **`TrackingModule.py`**: Implements `BallTracker`. Responsible for extracting the ball's center  and radius from video frames, handling resolution scaling between compressed video and calibration images.
* **`TrajectoryModule.py`**: The physics optimization engine. Contains `TrajectoryEstimator`, which uses `scipy.optimize.least_squares` to fit the 3D parabolic model to the 2D tracks, enforcing physical and semantic constraints.

---


### Pipeline Steps (Under the Hood)

1. **Calibration:** The system loads the camera model derived from manual PnP calibration on court landmarks.
2. **Shooter Detection:** YOLOv8-Pose identifies the player's feet in the first frame. These pixels are back-projected to the ground plane () to fix the trajectory start point ().
3. **Ball Tracking:** The ball is tracked in 2D. Coordinates are scaled to match the calibration resolution.
4. **Trajectory Fitting:** The optimizer solves for the 6 parameters  that minimize reprojection error while satisfying physical constraints (gravity, floor, rim height).

---

## Results

The system achieves high-fidelity reconstruction, validating the camera model with a spatial error of **< 3cm** on ground plane back-projection.

| Metric | Value |
| --- | --- |
| **Camera Height** | 1.58 m (Consistent with handheld setup) |
| **Distance to Hoop** | 12.75 m |
| **Physical Model** | Gravity  |

**Visual Output:**
The output video displays the original shot with the tracked ball (red) and the shooter's position (blue cross), synchronized with a 3D view of the canonical court.

---

## Acknowledgments

Developed at **Politecnico di Milano** for the Image Analysis and Computer Vision course.

**Authors:**

* Marcelo Takayama Russo
* Mateus Matzkin Gurza
* Julius Becker
