"""
Facial Expression Analyzer for Fatigue Detection

This module uses computer vision to analyze facial expressions and detect
signs of fatigue including eye closure, yawning, head pose changes, and
micro-expressions.
"""

import cv2
import numpy as np
import mediapipe as mp
import dlib
import time
import threading
import logging
from collections import deque
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import math

@dataclass
class FacialFrame:
    """Represents facial analysis data for a single frame"""
    timestamp: float
    landmarks: Optional[np.ndarray]
    eye_aspect_ratio: float
    mouth_aspect_ratio: float
    head_pose: Tuple[float, float, float]  # pitch, yaw, roll
    blink_detected: bool
    yawn_detected: bool
    drowsiness_score: float

class FacialAnalyzer:
    """Analyzes facial expressions for fatigue detection"""
    
    def __init__(self, camera_index: int = 0, fps: int = 10):
        self.camera_index = camera_index
        self.fps = fps
        self.frame_interval = 1.0 / fps
        
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Face mesh model
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize dlib face detector and predictor
        self.detector = dlib.get_frontal_face_detector()
        try:
            # Download shape predictor if not available
            self.predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
        except:
            logging.warning("dlib shape predictor not found. Using MediaPipe only.")
            self.predictor = None
        
        # Video capture
        self.cap = None
        self.is_monitoring = False
        
        # Data storage
        self.frame_buffer = deque(maxlen=300)  # Store last 30 seconds at 10 FPS
        self.analysis_window = deque(maxlen=100)  # Analysis window
        
        # Fatigue detection parameters
        self.eye_ar_threshold = 0.25
        self.eye_ar_consecutive_frames = 3
        self.yawn_threshold = 0.6
        self.drowsiness_threshold = 0.7
        
        # Counters and trackers
        self.blink_counter = 0
        self.yawn_counter = 0
        self.eye_closed_frames = 0
        self.total_frames = 0
        
        # Feature cache
        self.feature_cache = {}
        self.last_feature_update = 0
        
        # Threading
        self.capture_thread = None
        self.analysis_thread = None
        self.stop_event = threading.Event()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def start_monitoring(self):
        """Start facial monitoring"""
        if self.is_monitoring:
            return
        
        # Initialize camera
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_index}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        self.is_monitoring = True
        self.stop_event.clear()
        
        # Start capture and analysis threads
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.analysis_thread = threading.Thread(target=self._analysis_loop)
        
        self.capture_thread.start()
        self.analysis_thread.start()
        
        self.logger.info("Facial monitoring started")
    
    def stop_monitoring(self):
        """Stop facial monitoring"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        self.stop_event.set()
        
        # Wait for threads to finish
        if self.capture_thread:
            self.capture_thread.join()
        if self.analysis_thread:
            self.analysis_thread.join()
        
        # Release camera
        if self.cap:
            self.cap.release()
        
        self.logger.info("Facial monitoring stopped")
    
    def _capture_loop(self):
        """Main capture loop running in separate thread"""
        last_frame_time = 0
        
        while not self.stop_event.is_set():
            current_time = time.time()
            
            # Control frame rate
            if current_time - last_frame_time < self.frame_interval:
                time.sleep(0.01)
                continue
            
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Store frame with timestamp
            self.frame_buffer.append((current_time, frame))
            last_frame_time = current_time
    
    def _analysis_loop(self):
        """Analysis loop running in separate thread"""
        while not self.stop_event.is_set():
            if not self.frame_buffer:
                time.sleep(0.1)
                continue
            
            try:
                # Get latest frame
                timestamp, frame = self.frame_buffer[-1]
                
                # Analyze frame
                facial_data = self._analyze_frame(frame, timestamp)
                
                if facial_data:
                    self.analysis_window.append(facial_data)
                    self.total_frames += 1
                
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage
                
            except Exception as e:
                self.logger.error(f"Error in analysis loop: {e}")
    
    def _analyze_frame(self, frame: np.ndarray, timestamp: float) -> Optional[FacialFrame]:
        """Analyze a single frame for facial features"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                return None
            
            # Get face landmarks
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = self._extract_landmarks(face_landmarks, frame.shape)
            
            # Calculate facial features
            ear = self._calculate_eye_aspect_ratio(landmarks)
            mar = self._calculate_mouth_aspect_ratio(landmarks)
            head_pose = self._calculate_head_pose(landmarks, frame.shape)
            
            # Detect blinks and yawns
            blink_detected = ear < self.eye_ar_threshold
            yawn_detected = mar > self.yawn_threshold
            
            # Update counters
            if blink_detected:
                self.eye_closed_frames += 1
            else:
                if self.eye_closed_frames >= self.eye_ar_consecutive_frames:
                    self.blink_counter += 1
                self.eye_closed_frames = 0
            
            if yawn_detected:
                self.yawn_counter += 1
            
            # Calculate drowsiness score
            drowsiness_score = self._calculate_drowsiness_score(ear, mar, head_pose)
            
            return FacialFrame(
                timestamp=timestamp,
                landmarks=landmarks,
                eye_aspect_ratio=ear,
                mouth_aspect_ratio=mar,
                head_pose=head_pose,
                blink_detected=blink_detected,
                yawn_detected=yawn_detected,
                drowsiness_score=drowsiness_score
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing frame: {e}")
            return None
    
    def _extract_landmarks(self, face_landmarks, frame_shape) -> np.ndarray:
        """Extract landmark coordinates from MediaPipe results"""
        height, width = frame_shape[:2]
        landmarks = []
        
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            landmarks.append([x, y])
        
        return np.array(landmarks)
    
    def _calculate_eye_aspect_ratio(self, landmarks: np.ndarray) -> float:
        """Calculate Eye Aspect Ratio (EAR) for blink detection"""
        try:
            # MediaPipe eye landmark indices
            # Left eye: 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
            # Right eye: 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
            
            # Simplified eye landmarks for EAR calculation
            left_eye = landmarks[[33, 133, 160, 158, 144, 153]]
            right_eye = landmarks[[362, 263, 387, 385, 373, 380]]
            
            # Calculate EAR for both eyes
            left_ear = self._eye_aspect_ratio(left_eye)
            right_ear = self._eye_aspect_ratio(right_eye)
            
            # Return average EAR
            return (left_ear + right_ear) / 2.0
            
        except Exception as e:
            self.logger.error(f"Error calculating EAR: {e}")
            return 0.3  # Default value
    
    def _eye_aspect_ratio(self, eye_landmarks: np.ndarray) -> float:
        """Calculate EAR for a single eye"""
        try:
            # Vertical eye landmarks
            A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
            B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
            
            # Horizontal eye landmark
            C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
            
            # Calculate EAR
            ear = (A + B) / (2.0 * C)
            return ear
            
        except:
            return 0.3
    
    def _calculate_mouth_aspect_ratio(self, landmarks: np.ndarray) -> float:
        """Calculate Mouth Aspect Ratio (MAR) for yawn detection"""
        try:
            # Mouth landmarks (simplified)
            # Upper lip: 13, 14, 15
            # Lower lip: 17, 18, 19
            # Mouth corners: 61, 291
            
            mouth_top = landmarks[13]
            mouth_bottom = landmarks[17]
            mouth_left = landmarks[61]
            mouth_right = landmarks[291]
            
            # Vertical distance
            vertical_dist = np.linalg.norm(mouth_top - mouth_bottom)
            
            # Horizontal distance
            horizontal_dist = np.linalg.norm(mouth_left - mouth_right)
            
            # Calculate MAR
            mar = vertical_dist / horizontal_dist if horizontal_dist > 0 else 0
            return mar
            
        except Exception as e:
            self.logger.error(f"Error calculating MAR: {e}")
            return 0.3  # Default value
    
    def _calculate_head_pose(self, landmarks: np.ndarray, frame_shape) -> Tuple[float, float, float]:
        """Calculate head pose (pitch, yaw, roll)"""
        try:
            height, width = frame_shape[:2]
            
            # 3D model points (generic face model)
            model_points = np.array([
                (0.0, 0.0, 0.0),             # Nose tip
                (0.0, -330.0, -65.0),        # Chin
                (-225.0, 170.0, -135.0),     # Left eye left corner
                (225.0, 170.0, -135.0),      # Right eye right corner
                (-150.0, -150.0, -125.0),    # Left mouth corner
                (150.0, -150.0, -125.0)      # Right mouth corner
            ])
            
            # 2D image points from landmarks
            image_points = np.array([
                landmarks[1],      # Nose tip
                landmarks[152],    # Chin
                landmarks[33],     # Left eye left corner
                landmarks[362],    # Right eye right corner
                landmarks[61],     # Left mouth corner
                landmarks[291]     # Right mouth corner
            ], dtype=np.float64)
            
            # Camera matrix (approximate)
            focal_length = width
            center = (width / 2, height / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
            
            # Distortion coefficients (assume no distortion)
            dist_coeffs = np.zeros((4, 1))
            
            # Solve PnP
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs
            )
            
            if success:
                # Convert rotation vector to Euler angles
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                angles = self._rotation_matrix_to_euler_angles(rotation_matrix)
                return angles
            else:
                return (0.0, 0.0, 0.0)
                
        except Exception as e:
            self.logger.error(f"Error calculating head pose: {e}")
            return (0.0, 0.0, 0.0)
    
    def _rotation_matrix_to_euler_angles(self, R: np.ndarray) -> Tuple[float, float, float]:
        """Convert rotation matrix to Euler angles"""
        try:
            sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
            
            singular = sy < 1e-6
            
            if not singular:
                x = math.atan2(R[2, 1], R[2, 2])
                y = math.atan2(-R[2, 0], sy)
                z = math.atan2(R[1, 0], R[0, 0])
            else:
                x = math.atan2(-R[1, 2], R[1, 1])
                y = math.atan2(-R[2, 0], sy)
                z = 0
            
            # Convert to degrees
            pitch = math.degrees(x)
            yaw = math.degrees(y)
            roll = math.degrees(z)
            
            return (pitch, yaw, roll)
            
        except:
            return (0.0, 0.0, 0.0)
    
    def _calculate_drowsiness_score(self, ear: float, mar: float, head_pose: Tuple[float, float, float]) -> float:
        """Calculate overall drowsiness score"""
        try:
            # EAR component (lower EAR indicates more drowsiness)
            ear_score = max(0, (self.eye_ar_threshold - ear) / self.eye_ar_threshold)
            
            # MAR component (higher MAR indicates yawning)
            mar_score = min(1.0, mar / self.yawn_threshold)
            
            # Head pose component (head dropping indicates drowsiness)
            pitch, yaw, roll = head_pose
            head_score = min(1.0, max(0, (abs(pitch) - 10) / 30))  # Significant head tilt
            
            # Combine scores
            drowsiness_score = (ear_score * 0.5 + mar_score * 0.3 + head_score * 0.2)
            
            return min(1.0, drowsiness_score)
            
        except:
            return 0.0
    
    def extract_features(self) -> Dict[str, float]:
        """Extract facial features for fatigue detection"""
        current_time = time.time()
        
        # Use cached features if recent enough
        if (current_time - self.last_feature_update < 1.0 and self.feature_cache):
            return self.feature_cache
        
        if len(self.analysis_window) < 10:
            return self._get_default_features()
        
        # Get recent data (last 30 seconds)
        recent_data = [frame for frame in self.analysis_window 
                      if current_time - frame.timestamp <= 30]
        
        if not recent_data:
            return self._get_default_features()
        
        features = {}
        
        # Eye-related features
        features.update(self._extract_eye_features(recent_data))
        
        # Mouth-related features
        features.update(self._extract_mouth_features(recent_data))
        
        # Head pose features
        features.update(self._extract_head_pose_features(recent_data))
        
        # Temporal features
        features.update(self._extract_temporal_features(recent_data))
        
        # Overall drowsiness features
        features.update(self._extract_drowsiness_features(recent_data))
        
        self.feature_cache = features
        self.last_feature_update = current_time
        
        return features
    
    def _extract_eye_features(self, data: List[FacialFrame]) -> Dict[str, float]:
        """Extract eye-related features"""
        ear_values = [frame.eye_aspect_ratio for frame in data]
        blinks = [frame.blink_detected for frame in data]
        
        # Calculate blink rate (blinks per minute)
        time_span = max(data[-1].timestamp - data[0].timestamp, 1.0)
        blink_rate = (sum(blinks) / time_span) * 60
        
        # Eye closure duration
        closed_frames = sum(blinks)
        closure_ratio = closed_frames / len(data)
        
        return {
            'avg_eye_aspect_ratio': np.mean(ear_values),
            'eye_aspect_ratio_std': np.std(ear_values),
            'blink_rate': blink_rate,
            'eye_closure_ratio': closure_ratio,
            'min_eye_aspect_ratio': np.min(ear_values),
            'eye_fatigue_score': max(0, 1.0 - (np.mean(ear_values) / 0.3))
        }
    
    def _extract_mouth_features(self, data: List[FacialFrame]) -> Dict[str, float]:
        """Extract mouth-related features"""
        mar_values = [frame.mouth_aspect_ratio for frame in data]
        yawns = [frame.yawn_detected for frame in data]
        
        # Calculate yawn frequency
        time_span = max(data[-1].timestamp - data[0].timestamp, 1.0)
        yawn_frequency = (sum(yawns) / time_span) * 60
        
        return {
            'avg_mouth_aspect_ratio': np.mean(mar_values),
            'mouth_aspect_ratio_std': np.std(mar_values),
            'yawn_frequency': yawn_frequency,
            'max_mouth_aspect_ratio': np.max(mar_values),
            'yawn_fatigue_score': min(1.0, yawn_frequency / 5.0)  # 5 yawns per minute = max fatigue
        }
    
    def _extract_head_pose_features(self, data: List[FacialFrame]) -> Dict[str, float]:
        """Extract head pose features"""
        pitches = [frame.head_pose[0] for frame in data]
        yaws = [frame.head_pose[1] for frame in data]
        rolls = [frame.head_pose[2] for frame in data]
        
        # Head stability (lower std = more stable)
        pitch_stability = 1.0 / (1.0 + np.std(pitches))
        yaw_stability = 1.0 / (1.0 + np.std(yaws))
        roll_stability = 1.0 / (1.0 + np.std(rolls))
        
        # Head dropping (negative pitch indicates head dropping)
        avg_pitch = np.mean(pitches)
        head_drop_score = max(0, -avg_pitch / 30.0)  # Normalize to 0-1
        
        return {
            'avg_head_pitch': avg_pitch,
            'avg_head_yaw': np.mean(yaws),
            'avg_head_roll': np.mean(rolls),
            'head_pitch_stability': pitch_stability,
            'head_yaw_stability': yaw_stability,
            'head_roll_stability': roll_stability,
            'head_drop_score': head_drop_score,
            'head_movement_fatigue': 1.0 - ((pitch_stability + yaw_stability + roll_stability) / 3.0)
        }
    
    def _extract_temporal_features(self, data: List[FacialFrame]) -> Dict[str, float]:
        """Extract temporal pattern features"""
        # Analyze patterns over time
        drowsiness_scores = [frame.drowsiness_score for frame in data]
        
        # Trend analysis (increasing drowsiness over time)
        if len(drowsiness_scores) >= 10:
            x = np.arange(len(drowsiness_scores))
            slope, _ = np.polyfit(x, drowsiness_scores, 1)
            drowsiness_trend = max(0, slope * 100)  # Normalize
        else:
            drowsiness_trend = 0.0
        
        # Variability in drowsiness
        drowsiness_variability = np.std(drowsiness_scores)
        
        return {
            'drowsiness_trend': drowsiness_trend,
            'drowsiness_variability': drowsiness_variability,
            'avg_drowsiness_score': np.mean(drowsiness_scores),
            'max_drowsiness_score': np.max(drowsiness_scores)
        }
    
    def _extract_drowsiness_features(self, data: List[FacialFrame]) -> Dict[str, float]:
        """Extract overall drowsiness features"""
        drowsiness_scores = [frame.drowsiness_score for frame in data]
        
        # Percentage of time in drowsy state
        drowsy_frames = sum(1 for score in drowsiness_scores if score > self.drowsiness_threshold)
        drowsy_percentage = drowsy_frames / len(data)
        
        # Sustained drowsiness (consecutive drowsy frames)
        max_consecutive_drowsy = 0
        current_consecutive = 0
        
        for score in drowsiness_scores:
            if score > self.drowsiness_threshold:
                current_consecutive += 1
                max_consecutive_drowsy = max(max_consecutive_drowsy, current_consecutive)
            else:
                current_consecutive = 0
        
        return {
            'drowsy_percentage': drowsy_percentage,
            'max_consecutive_drowsy': max_consecutive_drowsy,
            'sustained_drowsiness_score': min(1.0, max_consecutive_drowsy / 30.0)  # 30 frames = high fatigue
        }
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default feature values when insufficient data"""
        return {
            'avg_eye_aspect_ratio': 0.3,
            'eye_aspect_ratio_std': 0.0,
            'blink_rate': 15.0,
            'eye_closure_ratio': 0.0,
            'min_eye_aspect_ratio': 0.3,
            'eye_fatigue_score': 0.0,
            'avg_mouth_aspect_ratio': 0.3,
            'mouth_aspect_ratio_std': 0.0,
            'yawn_frequency': 0.0,
            'max_mouth_aspect_ratio': 0.3,
            'yawn_fatigue_score': 0.0,
            'avg_head_pitch': 0.0,
            'avg_head_yaw': 0.0,
            'avg_head_roll': 0.0,
            'head_pitch_stability': 1.0,
            'head_yaw_stability': 1.0,
            'head_roll_stability': 1.0,
            'head_drop_score': 0.0,
            'head_movement_fatigue': 0.0,
            'drowsiness_trend': 0.0,
            'drowsiness_variability': 0.0,
            'avg_drowsiness_score': 0.0,
            'max_drowsiness_score': 0.0,
            'drowsy_percentage': 0.0,
            'max_consecutive_drowsy': 0.0,
            'sustained_drowsiness_score': 0.0
        }
    
    def get_fatigue_indicators(self) -> Dict[str, float]:
        """Get fatigue-specific indicators from facial analysis"""
        features = self.extract_features()
        
        indicators = {}
        
        # Eye fatigue (based on EAR and blink patterns)
        indicators['eye_fatigue'] = features['eye_fatigue_score']
        
        # Yawn fatigue
        indicators['yawn_fatigue'] = features['yawn_fatigue_score']
        
        # Head movement fatigue
        indicators['head_fatigue'] = features['head_movement_fatigue']
        
        # Sustained drowsiness
        indicators['sustained_fatigue'] = features['sustained_drowsiness_score']
        
        # Overall facial fatigue (weighted combination)
        weights = {
            'eye_fatigue': 0.4,
            'yawn_fatigue': 0.25,
            'head_fatigue': 0.20,
            'sustained_fatigue': 0.15
        }
        
        overall_fatigue = sum(indicators[key] * weights[key] for key in weights)
        indicators['overall_fatigue'] = overall_fatigue
        
        return indicators
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the current frame for display"""
        if self.frame_buffer:
            return self.frame_buffer[-1][1]
        return None
    
    def get_session_summary(self) -> Dict:
        """Get summary of current facial analysis session"""
        if not self.analysis_window:
            return {}
        
        features = self.extract_features()
        indicators = self.get_fatigue_indicators()
        
        session_duration = time.time() - self.analysis_window[0].timestamp if self.analysis_window else 0
        
        return {
            'session_duration': session_duration,
            'total_frames_analyzed': len(self.analysis_window),
            'total_blinks': self.blink_counter,
            'total_yawns': self.yawn_counter,
            'current_features': features,
            'fatigue_indicators': indicators,
            'timestamp': datetime.now().isoformat()
        }

if __name__ == "__main__":
    # Example usage
    analyzer = FacialAnalyzer()
    
    try:
        analyzer.start_monitoring()
        print("Facial analysis started. Look at the camera...")
        
        # Monitor for 60 seconds
        time.sleep(60)
        
        # Get results
        features = analyzer.extract_features()
        indicators = analyzer.get_fatigue_indicators()
        summary = analyzer.get_session_summary()
        
        print("\nFacial Features:")
        for key, value in features.items():
            print(f"  {key}: {value:.3f}")
        
        print("\nFatigue Indicators:")
        for key, value in indicators.items():
            print(f"  {key}: {value:.3f}")
        
        print(f"\nOverall Facial Fatigue Score: {indicators['overall_fatigue']:.3f}")
        
    except KeyboardInterrupt:
        print("\nStopping analysis...")
    finally:
        analyzer.stop_monitoring()