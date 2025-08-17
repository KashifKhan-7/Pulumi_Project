"""
AI Models for Fatigue Detection

This module implements machine learning models for predicting fatigue
from typing patterns, facial expressions, and wearable data features.
"""

import numpy as np
import pandas as pd
import pickle
import joblib
import logging
import os
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ModelPrediction:
    """Represents a model prediction result"""
    fatigue_probability: float
    fatigue_level: str  # 'low', 'moderate', 'high'
    confidence: float
    timestamp: datetime
    features_used: List[str]

@dataclass
class ModelPerformance:
    """Represents model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    cross_val_scores: List[float]

class FatigueClassifier:
    """Base class for fatigue classification models"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        self.performance_metrics = None
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def prepare_features(self, features: Dict[str, float]) -> np.ndarray:
        """Prepare features for model input"""
        if not self.feature_names:
            self.feature_names = sorted(features.keys())
        
        # Ensure all required features are present
        feature_vector = []
        for feature_name in self.feature_names:
            if feature_name in features:
                feature_vector.append(features[feature_name])
            else:
                feature_vector.append(0.0)  # Default value for missing features
        
        return np.array(feature_vector).reshape(1, -1)
    
    def predict(self, features: Dict[str, float]) -> ModelPrediction:
        """Make a prediction using the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare features
        X = self.prepare_features(features)
        X_scaled = self.scaler.transform(X)
        
        # Get prediction and probability
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        # Convert to fatigue probability (assuming binary classification: 0=not fatigued, 1=fatigued)
        fatigue_prob = probabilities[1] if len(probabilities) > 1 else probabilities[0]
        
        # Determine fatigue level
        if fatigue_prob < 0.3:
            fatigue_level = 'low'
        elif fatigue_prob < 0.7:
            fatigue_level = 'moderate'
        else:
            fatigue_level = 'high'
        
        # Calculate confidence (distance from decision boundary)
        confidence = max(probabilities)
        
        return ModelPrediction(
            fatigue_probability=fatigue_prob,
            fatigue_level=fatigue_level,
            confidence=confidence,
            timestamp=datetime.now(),
            features_used=self.feature_names
        )
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_name': self.model_name,
            'performance_metrics': self.performance_metrics
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        if not os.path.exists(filepath):
            self.logger.warning(f"Model file not found: {filepath}")
            return False
        
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.performance_metrics = model_data.get('performance_metrics')
            self.is_trained = True
            self.logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False

class RandomForestFatigueClassifier(FatigueClassifier):
    """Random Forest classifier for fatigue detection"""
    
    def __init__(self):
        super().__init__("RandomForest")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
        """Train the Random Forest model"""
        self.feature_names = feature_names
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate performance
        self.performance_metrics = self._evaluate_model(X_test_scaled, y_test)
        self.is_trained = True
        
        self.logger.info(f"Random Forest model trained. Accuracy: {self.performance_metrics.accuracy:.3f}")
        return self.performance_metrics
    
    def _evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> ModelPerformance:
        """Evaluate model performance"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_test, y_test, cv=5)
        
        return ModelPerformance(
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred, average='weighted'),
            recall=recall_score(y_test, y_pred, average='weighted'),
            f1_score=f1_score(y_test, y_pred, average='weighted'),
            auc_score=roc_auc_score(y_test, y_pred_proba),
            cross_val_scores=cv_scores.tolist()
        )
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if not self.is_trained:
            return {}
        
        importance_scores = self.model.feature_importances_
        return dict(zip(self.feature_names, importance_scores))

class GradientBoostingFatigueClassifier(FatigueClassifier):
    """Gradient Boosting classifier for fatigue detection"""
    
    def __init__(self):
        super().__init__("GradientBoosting")
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
        """Train the Gradient Boosting model"""
        self.feature_names = feature_names
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate performance
        self.performance_metrics = self._evaluate_model(X_test_scaled, y_test)
        self.is_trained = True
        
        self.logger.info(f"Gradient Boosting model trained. Accuracy: {self.performance_metrics.accuracy:.3f}")
        return self.performance_metrics
    
    def _evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> ModelPerformance:
        """Evaluate model performance"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_test, y_test, cv=5)
        
        return ModelPerformance(
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred, average='weighted'),
            recall=recall_score(y_test, y_pred, average='weighted'),
            f1_score=f1_score(y_test, y_pred, average='weighted'),
            auc_score=roc_auc_score(y_test, y_pred_proba),
            cross_val_scores=cv_scores.tolist()
        )

class NeuralNetworkFatigueClassifier(FatigueClassifier):
    """Neural Network classifier for fatigue detection"""
    
    def __init__(self):
        super().__init__("NeuralNetwork")
        self.model = None
    
    def _build_model(self, input_dim: int):
        """Build the neural network architecture"""
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
        """Train the Neural Network model"""
        self.feature_names = feature_names
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Build and train model
        self.model = self._build_model(X_train_scaled.shape[1])
        
        # Train with early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate performance
        self.performance_metrics = self._evaluate_model(X_test_scaled, y_test)
        self.is_trained = True
        
        self.logger.info(f"Neural Network model trained. Accuracy: {self.performance_metrics.accuracy:.3f}")
        return self.performance_metrics
    
    def predict(self, features: Dict[str, float]) -> ModelPrediction:
        """Make a prediction using the trained neural network"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare features
        X = self.prepare_features(features)
        X_scaled = self.scaler.transform(X)
        
        # Get prediction probability
        fatigue_prob = self.model.predict(X_scaled, verbose=0)[0][0]
        
        # Determine fatigue level
        if fatigue_prob < 0.3:
            fatigue_level = 'low'
        elif fatigue_prob < 0.7:
            fatigue_level = 'moderate'
        else:
            fatigue_level = 'high'
        
        # Confidence is the probability itself for neural networks
        confidence = max(fatigue_prob, 1 - fatigue_prob)
        
        return ModelPrediction(
            fatigue_probability=fatigue_prob,
            fatigue_level=fatigue_level,
            confidence=confidence,
            timestamp=datetime.now(),
            features_used=self.feature_names
        )
    
    def _evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> ModelPerformance:
        """Evaluate neural network performance"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        y_pred_proba = self.model.predict(X_test, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        return ModelPerformance(
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred, average='weighted'),
            recall=recall_score(y_test, y_pred, average='weighted'),
            f1_score=f1_score(y_test, y_pred, average='weighted'),
            auc_score=roc_auc_score(y_test, y_pred_proba),
            cross_val_scores=[]  # Cross-validation not implemented for neural networks
        )

class DataGenerator:
    """Generate synthetic training data for fatigue detection models"""
    
    def __init__(self, n_samples: int = 10000):
        self.n_samples = n_samples
        self.logger = logging.getLogger(__name__)
    
    def generate_training_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate synthetic training data with realistic fatigue patterns"""
        np.random.seed(42)
        
        data = []
        labels = []
        
        for i in range(self.n_samples):
            # Generate base fatigue level (0 = not fatigued, 1 = fatigued)
            fatigue_level = np.random.choice([0, 1], p=[0.6, 0.4])  # 40% fatigued samples
            
            # Generate features based on fatigue level
            sample = self._generate_sample_features(fatigue_level)
            
            data.append(sample)
            labels.append(fatigue_level)
        
        # Create feature names
        feature_names = self._get_feature_names()
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=feature_names)
        y = np.array(labels)
        
        self.logger.info(f"Generated {self.n_samples} training samples")
        self.logger.info(f"Fatigue distribution: {np.bincount(y)}")
        
        return df, y
    
    def _generate_sample_features(self, fatigue_level: int) -> List[float]:
        """Generate feature values for a single sample"""
        features = []
        
        # Typing features (18 features)
        if fatigue_level == 1:  # Fatigued
            # Slower typing, more errors, irregular rhythm
            features.extend([
                np.random.normal(0.8, 0.2),   # avg_keystroke_interval (higher = slower)
                np.random.normal(0.3, 0.1),   # keystroke_interval_std (higher = more irregular)
                np.random.normal(0.4, 0.1),   # keystroke_interval_cv
                np.random.normal(25, 8),      # typing_speed_wpm (lower = slower)
                np.random.normal(0.4, 0.1),   # typing_speed_cps
                np.random.normal(0.3, 0.1),   # rhythm_regularity (lower = worse)
                np.random.normal(8, 3),       # flow_interruptions
                np.random.normal(0.4, 0.1),   # pause_frequency
                np.random.normal(0.3, 0.1),   # long_pause_ratio
                np.random.normal(0.15, 0.05), # error_rate (higher = more errors)
                np.random.normal(12, 4),      # correction_frequency
                np.random.normal(5, 2),       # correction_bursts
                np.random.normal(0.15, 0.05), # avg_key_duration
                np.random.normal(0.08, 0.02), # key_duration_std
                np.random.normal(0.01, 0.005), # pressure_variance
                np.random.normal(120, 30),    # activity_level
                np.random.normal(0.4, 0.1),   # typing_consistency
                np.random.normal(0.2, 0.1),   # micro_break_frequency
            ])
        else:  # Not fatigued
            # Faster typing, fewer errors, regular rhythm
            features.extend([
                np.random.normal(0.3, 0.1),   # avg_keystroke_interval
                np.random.normal(0.1, 0.05),  # keystroke_interval_std
                np.random.normal(0.2, 0.05),  # keystroke_interval_cv
                np.random.normal(55, 10),     # typing_speed_wpm
                np.random.normal(0.9, 0.2),   # typing_speed_cps
                np.random.normal(0.8, 0.1),   # rhythm_regularity
                np.random.normal(2, 1),       # flow_interruptions
                np.random.normal(0.1, 0.05),  # pause_frequency
                np.random.normal(0.05, 0.02), # long_pause_ratio
                np.random.normal(0.03, 0.02), # error_rate
                np.random.normal(3, 2),       # correction_frequency
                np.random.normal(1, 1),       # correction_bursts
                np.random.normal(0.08, 0.02), # avg_key_duration
                np.random.normal(0.03, 0.01), # key_duration_std
                np.random.normal(0.002, 0.001), # pressure_variance
                np.random.normal(200, 40),    # activity_level
                np.random.normal(0.8, 0.1),   # typing_consistency
                np.random.normal(0.05, 0.03), # micro_break_frequency
            ])
        
        # Facial features (23 features)
        if fatigue_level == 1:  # Fatigued
            features.extend([
                np.random.normal(0.18, 0.05), # avg_eye_aspect_ratio (lower = more closed)
                np.random.normal(0.08, 0.02), # eye_aspect_ratio_std
                np.random.normal(8, 3),       # blink_rate (lower when tired)
                np.random.normal(0.4, 0.1),   # eye_closure_ratio
                np.random.normal(0.12, 0.03), # min_eye_aspect_ratio
                np.random.normal(0.7, 0.2),   # eye_fatigue_score
                np.random.normal(0.8, 0.2),   # avg_mouth_aspect_ratio (higher = yawning)
                np.random.normal(0.3, 0.1),   # mouth_aspect_ratio_std
                np.random.normal(4, 2),       # yawn_frequency
                np.random.normal(1.2, 0.3),   # max_mouth_aspect_ratio
                np.random.normal(0.8, 0.2),   # yawn_fatigue_score
                np.random.normal(-15, 8),     # avg_head_pitch (negative = dropping)
                np.random.normal(5, 5),       # avg_head_yaw
                np.random.normal(3, 3),       # avg_head_roll
                np.random.normal(0.3, 0.1),   # head_pitch_stability
                np.random.normal(0.4, 0.1),   # head_yaw_stability
                np.random.normal(0.5, 0.1),   # head_roll_stability
                np.random.normal(0.5, 0.2),   # head_drop_score
                np.random.normal(0.6, 0.2),   # head_movement_fatigue
                np.random.normal(0.02, 0.01), # drowsiness_trend
                np.random.normal(0.3, 0.1),   # drowsiness_variability
                np.random.normal(0.7, 0.2),   # avg_drowsiness_score
                np.random.normal(0.9, 0.1),   # max_drowsiness_score
                np.random.normal(0.6, 0.2),   # drowsy_percentage
                np.random.normal(15, 8),      # max_consecutive_drowsy
                np.random.normal(0.5, 0.2),   # sustained_drowsiness_score
            ])
        else:  # Not fatigued
            features.extend([
                np.random.normal(0.28, 0.05), # avg_eye_aspect_ratio
                np.random.normal(0.04, 0.01), # eye_aspect_ratio_std
                np.random.normal(18, 5),      # blink_rate
                np.random.normal(0.1, 0.05),  # eye_closure_ratio
                np.random.normal(0.22, 0.03), # min_eye_aspect_ratio
                np.random.normal(0.1, 0.1),   # eye_fatigue_score
                np.random.normal(0.3, 0.1),   # avg_mouth_aspect_ratio
                np.random.normal(0.1, 0.05),  # mouth_aspect_ratio_std
                np.random.normal(0.5, 0.5),   # yawn_frequency
                np.random.normal(0.5, 0.1),   # max_mouth_aspect_ratio
                np.random.normal(0.1, 0.1),   # yawn_fatigue_score
                np.random.normal(2, 5),       # avg_head_pitch
                np.random.normal(0, 5),       # avg_head_yaw
                np.random.normal(0, 3),       # avg_head_roll
                np.random.normal(0.8, 0.1),   # head_pitch_stability
                np.random.normal(0.8, 0.1),   # head_yaw_stability
                np.random.normal(0.9, 0.1),   # head_roll_stability
                np.random.normal(0.1, 0.1),   # head_drop_score
                np.random.normal(0.2, 0.1),   # head_movement_fatigue
                np.random.normal(-0.01, 0.01), # drowsiness_trend
                np.random.normal(0.1, 0.05),  # drowsiness_variability
                np.random.normal(0.2, 0.1),   # avg_drowsiness_score
                np.random.normal(0.4, 0.2),   # max_drowsiness_score
                np.random.normal(0.1, 0.1),   # drowsy_percentage
                np.random.normal(2, 3),       # max_consecutive_drowsy
                np.random.normal(0.07, 0.1),  # sustained_drowsiness_score
            ])
        
        # Wearable features (26 features)
        if fatigue_level == 1:  # Fatigued
            features.extend([
                np.random.normal(85, 10),     # avg_heart_rate (higher when fatigued)
                np.random.normal(15, 5),      # heart_rate_std
                np.random.normal(0.5, 0.3),   # heart_rate_trend
                np.random.normal(30, 8),      # avg_hrv (lower when fatigued)
                np.random.normal(8, 3),       # hrv_std
                np.random.normal(-0.5, 0.3),  # hrv_trend (decreasing)
                np.random.normal(80, 8),      # resting_heart_rate
                np.random.normal(0.4, 0.2),   # heart_rate_recovery
                np.random.normal(15, 8),      # avg_activity_level (lower)
                np.random.normal(8, 3),       # activity_std
                np.random.normal(-0.2, 0.1),  # activity_trend
                np.random.normal(500, 200),   # total_steps
                np.random.normal(8, 4),       # steps_per_minute
                np.random.normal(50, 20),     # total_calories
                np.random.normal(0.7, 0.2),   # sedentary_time_ratio (higher)
                np.random.normal(0.3, 0.1),   # activity_consistency
                np.random.normal(65, 10),     # avg_sleep_score (lower)
                np.random.normal(12, 4),      # sleep_score_std
                np.random.normal(-0.3, 0.2),  # sleep_trend
                np.random.normal(0.4, 0.1),   # sleep_consistency
                np.random.normal(0.6, 0.2),   # recovery_score
                np.random.normal(65, 15),     # avg_stress_level (higher)
                np.random.normal(20, 8),      # stress_std
                np.random.normal(0.3, 0.2),   # stress_trend
                np.random.normal(0.4, 0.2),   # high_stress_ratio
                np.random.normal(0.8, 0.3),   # stress_variability
                np.random.normal(37.2, 0.5),  # avg_skin_temperature
                np.random.normal(0.4, 0.1),   # temperature_std
                np.random.normal(0.01, 0.01), # temperature_trend
                np.random.normal(96, 2),      # avg_blood_oxygen (lower)
                np.random.normal(1.5, 0.5),   # blood_oxygen_std
                np.random.normal(0.1, 0.1),   # low_oxygen_ratio
            ])
        else:  # Not fatigued
            features.extend([
                np.random.normal(70, 8),      # avg_heart_rate
                np.random.normal(8, 3),       # heart_rate_std
                np.random.normal(0, 0.2),     # heart_rate_trend
                np.random.normal(55, 10),     # avg_hrv
                np.random.normal(12, 4),      # hrv_std
                np.random.normal(0, 0.2),     # hrv_trend
                np.random.normal(65, 5),      # resting_heart_rate
                np.random.normal(0.9, 0.1),   # heart_rate_recovery
                np.random.normal(45, 15),     # avg_activity_level
                np.random.normal(15, 5),      # activity_std
                np.random.normal(0, 0.1),     # activity_trend
                np.random.normal(1200, 300),  # total_steps
                np.random.normal(20, 8),      # steps_per_minute
                np.random.normal(120, 30),    # total_calories
                np.random.normal(0.3, 0.1),   # sedentary_time_ratio
                np.random.normal(0.8, 0.1),   # activity_consistency
                np.random.normal(85, 8),      # avg_sleep_score
                np.random.normal(6, 2),       # sleep_score_std
                np.random.normal(0, 0.1),     # sleep_trend
                np.random.normal(0.9, 0.1),   # sleep_consistency
                np.random.normal(0.95, 0.05), # recovery_score
                np.random.normal(25, 10),     # avg_stress_level
                np.random.normal(8, 3),       # stress_std
                np.random.normal(0, 0.1),     # stress_trend
                np.random.normal(0.05, 0.05), # high_stress_ratio
                np.random.normal(0.3, 0.1),   # stress_variability
                np.random.normal(36.8, 0.3),  # avg_skin_temperature
                np.random.normal(0.2, 0.05),  # temperature_std
                np.random.normal(0, 0.005),   # temperature_trend
                np.random.normal(98, 1),      # avg_blood_oxygen
                np.random.normal(0.5, 0.2),   # blood_oxygen_std
                np.random.normal(0, 0.02),    # low_oxygen_ratio
            ])
        
        # Clip values to reasonable ranges
        features = np.clip(features, 0, None)  # No negative values
        
        return features
    
    def _get_feature_names(self) -> List[str]:
        """Get the names of all features"""
        typing_features = [
            'avg_keystroke_interval', 'keystroke_interval_std', 'keystroke_interval_cv',
            'typing_speed_wpm', 'typing_speed_cps', 'rhythm_regularity',
            'flow_interruptions', 'pause_frequency', 'long_pause_ratio',
            'error_rate', 'correction_frequency', 'correction_bursts',
            'avg_key_duration', 'key_duration_std', 'pressure_variance',
            'activity_level', 'typing_consistency', 'micro_break_frequency'
        ]
        
        facial_features = [
            'avg_eye_aspect_ratio', 'eye_aspect_ratio_std', 'blink_rate',
            'eye_closure_ratio', 'min_eye_aspect_ratio', 'eye_fatigue_score',
            'avg_mouth_aspect_ratio', 'mouth_aspect_ratio_std', 'yawn_frequency',
            'max_mouth_aspect_ratio', 'yawn_fatigue_score', 'avg_head_pitch',
            'avg_head_yaw', 'avg_head_roll', 'head_pitch_stability',
            'head_yaw_stability', 'head_roll_stability', 'head_drop_score',
            'head_movement_fatigue', 'drowsiness_trend', 'drowsiness_variability',
            'avg_drowsiness_score', 'max_drowsiness_score', 'drowsy_percentage',
            'max_consecutive_drowsy', 'sustained_drowsiness_score'
        ]
        
        wearable_features = [
            'avg_heart_rate', 'heart_rate_std', 'heart_rate_trend',
            'avg_hrv', 'hrv_std', 'hrv_trend', 'resting_heart_rate',
            'heart_rate_recovery', 'avg_activity_level', 'activity_std',
            'activity_trend', 'total_steps', 'steps_per_minute',
            'total_calories', 'sedentary_time_ratio', 'activity_consistency',
            'avg_sleep_score', 'sleep_score_std', 'sleep_trend',
            'sleep_consistency', 'recovery_score', 'avg_stress_level',
            'stress_std', 'stress_trend', 'high_stress_ratio',
            'stress_variability', 'avg_skin_temperature', 'temperature_std',
            'temperature_trend', 'avg_blood_oxygen', 'blood_oxygen_std',
            'low_oxygen_ratio'
        ]
        
        return typing_features + facial_features + wearable_features

class ModelTrainer:
    """Train and manage multiple fatigue detection models"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        self.models = {
            'random_forest': RandomForestFatigueClassifier(),
            'gradient_boosting': GradientBoostingFatigueClassifier(),
            'neural_network': NeuralNetworkFatigueClassifier()
        }
        
        self.logger = logging.getLogger(__name__)
    
    def train_all_models(self, generate_data: bool = True):
        """Train all available models"""
        if generate_data:
            # Generate synthetic training data
            data_generator = DataGenerator(n_samples=10000)
            X_df, y = data_generator.generate_training_data()
            X = X_df.values
            feature_names = X_df.columns.tolist()
        else:
            raise NotImplementedError("Real data loading not implemented yet")
        
        results = {}
        
        for model_name, model in self.models.items():
            self.logger.info(f"Training {model_name} model...")
            
            try:
                performance = model.train(X, y, feature_names)
                results[model_name] = performance
                
                # Save trained model
                model_path = os.path.join(self.models_dir, f"{model_name}_fatigue_model.pkl")
                model.save_model(model_path)
                
                self.logger.info(f"{model_name} training completed. Accuracy: {performance.accuracy:.3f}")
                
            except Exception as e:
                self.logger.error(f"Error training {model_name}: {e}")
                results[model_name] = None
        
        return results
    
    def load_all_models(self):
        """Load all trained models"""
        for model_name, model in self.models.items():
            model_path = os.path.join(self.models_dir, f"{model_name}_fatigue_model.pkl")
            success = model.load_model(model_path)
            if not success:
                self.logger.warning(f"Could not load {model_name} model")
    
    def get_model(self, model_name: str) -> Optional[FatigueClassifier]:
        """Get a specific model"""
        return self.models.get(model_name)
    
    def predict_with_all_models(self, features: Dict[str, float]) -> Dict[str, ModelPrediction]:
        """Make predictions with all trained models"""
        predictions = {}
        
        for model_name, model in self.models.items():
            if model.is_trained:
                try:
                    prediction = model.predict(features)
                    predictions[model_name] = prediction
                except Exception as e:
                    self.logger.error(f"Error predicting with {model_name}: {e}")
            else:
                self.logger.warning(f"Model {model_name} is not trained")
        
        return predictions

if __name__ == "__main__":
    # Example usage
    trainer = ModelTrainer()
    
    # Train all models
    print("Training models...")
    results = trainer.train_all_models()
    
    # Print results
    print("\nTraining Results:")
    for model_name, performance in results.items():
        if performance:
            print(f"{model_name}:")
            print(f"  Accuracy: {performance.accuracy:.3f}")
            print(f"  F1 Score: {performance.f1_score:.3f}")
            print(f"  AUC Score: {performance.auc_score:.3f}")
    
    # Test predictions with sample features
    print("\nTesting predictions...")
    sample_features = {
        'avg_keystroke_interval': 0.5,
        'typing_speed_wpm': 30,
        'avg_eye_aspect_ratio': 0.2,
        'blink_rate': 10,
        'avg_heart_rate': 85,
        'avg_hrv': 35,
        'avg_sleep_score': 70,
        'avg_stress_level': 60
    }
    
    # Add all required features with default values
    data_generator = DataGenerator()
    all_feature_names = data_generator._get_feature_names()
    for feature_name in all_feature_names:
        if feature_name not in sample_features:
            sample_features[feature_name] = 0.5  # Default value
    
    predictions = trainer.predict_with_all_models(sample_features)
    
    print("\nPredictions:")
    for model_name, prediction in predictions.items():
        print(f"{model_name}: {prediction.fatigue_level} "
              f"(probability: {prediction.fatigue_probability:.3f}, "
              f"confidence: {prediction.confidence:.3f})")