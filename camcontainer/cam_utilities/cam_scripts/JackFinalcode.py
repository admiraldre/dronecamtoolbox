import cv2
import numpy as np
import threading
import sys
import os

# If aprilgrid is under ../src:
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "enhanced_python_aprilgrid", "src")))
from aprilgrid import Detector  

class CameraCalibration:
    def __init__(self, primary_intrinsics, secondary_intrinsics, primary_distortion, secondary_distortion, extrinsics):
        self.primary_intrinsics = primary_intrinsics
        self.secondary_intrinsics = secondary_intrinsics
        self.primary_distortion = primary_distortion
        self.secondary_distortion = secondary_distortion
        self.extrinsics = extrinsics
        self.R = extrinsics[:3, :3]
        self.T = extrinsics[:3, 3]

class StereoProcessor:
    def __init__(self):
        # Camera intrinsic parameters
        intrinsics_cam0 = np.array([3315.158744682164, 3317.4423656635427, 1044.9919174051424, 769.2791390484149])
        intrinsics_cam1 = np.array([3303.1303751889086, 3304.4918413538717, 1021.6391642423999, 738.8164474717523])

        # Transformation matrix between cameras
        extrinsics = np.array([
            [0.9958785322848419, -0.001883418317043302, 0.09067745954553587, -0.19167885361868625],
            [0.0021215517599516157, 0.9999945494395888, -0.0025298437192514665, -0.0007459795575273124],
            [-0.09067220054856458, 0.0027117939739249486, 0.9958770999581846, 0.006433227010261678],
            [0.0, 0.0, 0.0, 1.0]
        ])

        # Distortion coefficients
        distortion_coeffs_cam0 = np.array([-0.26727533865087455, 0.6914547210589541, -0.00022631332732631944, 0.0006445742114670336])
        distortion_coeffs_cam1 = np.array([-0.25984247075537714, 0.609867968623631, 0.0008865986409872509, -0.000975406239237423])

        self.calibration = CameraCalibration(intrinsics_cam0, intrinsics_cam1, distortion_coeffs_cam0, distortion_coeffs_cam1, extrinsics)

        self.image_size = (2012, 1518)

        # Intrinsic camera matrices
        self.K1 = np.array([
            [self.calibration.primary_intrinsics[0], 0, self.calibration.primary_intrinsics[2]],
            [0, self.calibration.primary_intrinsics[1], self.calibration.primary_intrinsics[3]],
            [0, 0, 1]
        ])
        
        self.K2 = np.array([
            [self.calibration.secondary_intrinsics[0], 0, self.calibration.secondary_intrinsics[2]],
            [0, self.calibration.secondary_intrinsics[1], self.calibration.secondary_intrinsics[3]],
            [0, 0, 1]
        ])

        # Stereo rectification
        self.R1_rect, self.R2_rect, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            self.K1, self.calibration.primary_distortion, self.K2, self.calibration.secondary_distortion, self.image_size, self.calibration.R, self.calibration.T, flags=cv2.CALIB_ZERO_DISPARITY
        )

        # Rectification maps
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(self.K1, self.calibration.primary_distortion, self.R1_rect, self.P1, self.image_size, cv2.CV_32F)
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(self.K2, self.calibration.secondary_distortion, self.R2_rect, self.P2, self.image_size, cv2.CV_32F)

        # Initialize AprilTag detector
        self.apriltag_detector = Detector('t16h5b1')

    def process_images(self, image_path_cam0, image_path_cam1):
        image_cam0 = cv2.imread(image_path_cam0)
        image_cam1 = cv2.imread(image_path_cam1)
        if image_cam0 is None or image_cam1 is None:
            print("Error: Could not load images.")
            return
        print("Images loaded successfully.")
        
        bin_factor = 2
        height, width = image_cam0.shape[:2]
        height, width = image_cam1.shape[:2]
        
        print(height, width) # this was because the pics where high res
        
        image_cam0 = cv2.resize(image_cam0, (width // bin_factor, height // bin_factor), interpolation=cv2.INTER_AREA)
        image_cam1 = cv2.resize(image_cam1, (width // bin_factor, height // bin_factor), interpolation=cv2.INTER_AREA)
        
        rectified_cam0 = cv2.remap(image_cam0, self.map1x, self.map1y, interpolation=cv2.INTER_LINEAR)
        rectified_cam1 = cv2.remap(image_cam1, self.map2x, self.map2y, interpolation=cv2.INTER_LINEAR)
        
        detections_left = self.detect_apriltags(rectified_cam0)
        detections_right = self.detect_apriltags(rectified_cam1)
        self.compute_3d_positions(detections_left, detections_right)

    def detect_apriltags(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = self.apriltag_detector.detect(gray)
        return {tag.tag_id: np.mean(np.array(tag.corners), axis=0).flatten() for tag in detections}

    def compute_3d_positions(self, detections_left, detections_right):
        matched_ids = set(detections_left.keys()).intersection(detections_right.keys())
        if not matched_ids:
            print("No matching AprilTags found in both images.")
            return
        left_pts = np.array([detections_left[tag_id] for tag_id in matched_ids])
        right_pts = np.array([detections_right[tag_id] for tag_id in matched_ids])
        points_4d_hom = cv2.triangulatePoints(self.P1, self.P2, left_pts.T, right_pts.T)
        points_3d = (points_4d_hom[:3] / points_4d_hom[3]).T
        for tag_id, (x, y, z) in zip(matched_ids, points_3d):
            print(f"Tag ID {tag_id}: 3D Position (X: {x:.6f}, Y: {y:.6f}, Z: {z:.6f})")

# Example usage
if __name__ == "__main__":
    processor = StereoProcessor()
    processor.process_images('camera_DEV_1AB22C00E123_imageT5_1.png', 'camera_DEV_1AB22C00E588_imageT5_1.png')
