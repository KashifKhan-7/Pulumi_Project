"""
Multi-Modal Fusion System for Fatigue Detection

This module combines data from typing patterns, facial expressions, and wearable
devices to provide comprehensive fatigue detection using various fusion strategies.
"""

import time
import threading
import numpy as np
import pandas as pd
import logging
import yaml
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque
import sqlite3
import json

# Import our modules
from typing_analyzer import TypingAnalyzer
from facial_analyzer import FacialAnalyzer
from wearable_analyzer import WearableAnalyzer
from ai_models import ModelTrainer, ModelPrediction

@dataclass
class FusionResult:
    """Represents the result of multi-modal fusion"""
    overall_fatigue_score: float
    fatigue_level: str  # 'low', 'moderate', 'high', 'critical'
    confidence: float
    timestamp: datetime
    
    # Individual modality scores
    typing_score: float
    facial_score: float
    wearable_score: float
    
    # Individual modality confidence
    typing_confidence: float
    facial_confidence: float
    wearable_confidence: float
    
    # Detailed predictions
    typing_prediction: Optional[Dict]
    facial_prediction: Optional[Dict]
    wearable_prediction: Optional[Dict]
    ml_predictions: Optional[Dict]
    
    # Alert information
    alert_level: str  # 'none', 'low', 'medium', 'high', 'critical'
    recommendations: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result

@dataclass
class SystemStatus:
    """Represents the current system status"""
    is_active: bool
    typing_active: bool
    facial_active: bool
    wearable_active: bool
    last_update: datetime
    total_sessions: int
    current_session_duration: float
    alerts_count: int

class FusionStrategy:
    """Base class for fusion strategies"""
    
    def __init__(self, name: str):
        self.name = name
    
    def fuse(self, typing_score: float, facial_score: float, wearable_score: float,
             typing_conf: float, facial_conf: float, wearable_conf: float) -> Tuple[float, float]:
        """Fuse scores from multiple modalities"""
        raise NotImplementedError

class WeightedAverageFusion(FusionStrategy):
    """Weighted average fusion strategy"""
    
    def __init__(self, weights: Dict[str, float]):
        super().__init__("WeightedAverage")
        self.weights = weights
        
        # Normalize weights
        total_weight = sum(weights.values())
        self.weights = {k: v/total_weight for k, v in weights.items()}
    
    def fuse(self, typing_score: float, facial_score: float, wearable_score: float,
             typing_conf: float, facial_conf: float, wearable_conf: float) -> Tuple[float, float]:
        """Weighted average fusion"""
        scores = [typing_score, facial_score, wearable_score]
        confidences = [typing_conf, facial_conf, wearable_conf]
        weight_keys = ['typing', 'facial', 'wearable']
        
        # Calculate weighted score
        weighted_score = sum(score * self.weights[key] 
                           for score, key in zip(scores, weight_keys))
        
        # Calculate weighted confidence
        weighted_confidence = sum(conf * self.weights[key] 
                                for conf, key in zip(confidences, weight_keys))
        
        return weighted_score, weighted_confidence

class ConfidenceWeightedFusion(FusionStrategy):
    """Confidence-weighted fusion strategy"""
    
    def __init__(self):
        super().__init__("ConfidenceWeighted")
    
    def fuse(self, typing_score: float, facial_score: float, wearable_score: float,
             typing_conf: float, facial_conf: float, wearable_conf: float) -> Tuple[float, float]:
        """Confidence-weighted fusion"""
        scores = np.array([typing_score, facial_score, wearable_score])
        confidences = np.array([typing_conf, facial_conf, wearable_conf])
        
        # Avoid division by zero
        confidences = np.maximum(confidences, 0.001)
        
        # Normalize confidence weights
        conf_weights = confidences / np.sum(confidences)
        
        # Calculate weighted score and confidence
        fused_score = np.sum(scores * conf_weights)
        fused_confidence = np.mean(confidences)
        
        return fused_score, fused_confidence

class AdaptiveFusion(FusionStrategy):
    """Adaptive fusion that adjusts weights based on historical performance"""
    
    def __init__(self, initial_weights: Dict[str, float]):
        super().__init__("Adaptive")
        self.weights = initial_weights.copy()
        self.performance_history = deque(maxlen=100)
        self.adaptation_rate = 0.01
    
    def fuse(self, typing_score: float, facial_score: float, wearable_score: float,
             typing_conf: float, facial_conf: float, wearable_conf: float) -> Tuple[float, float]:
        """Adaptive fusion with weight adjustment"""
        scores = np.array([typing_score, facial_score, wearable_score])
        confidences = np.array([typing_conf, facial_conf, wearable_conf])
        weight_keys = ['typing', 'facial', 'wearable']
        
        # Get current weights
        current_weights = np.array([self.weights[key] for key in weight_keys])
        
        # Calculate weighted score
        fused_score = np.sum(scores * current_weights)
        fused_confidence = np.sum(confidences * current_weights)
        
        # Update performance history
        self.performance_history.append({
            'scores': scores.copy(),
            'confidences': confidences.copy(),
            'weights': current_weights.copy(),
            'fused_score': fused_score
        })
        
        # Adapt weights based on recent performance
        self._adapt_weights()
        
        return fused_score, fused_confidence
    
    def _adapt_weights(self):
        """Adapt weights based on performance history"""
        if len(self.performance_history) < 10:
            return
        
        # Simple adaptation: increase weights of more confident modalities
        recent_data = list(self.performance_history)[-10:]
        avg_confidences = np.mean([data['confidences'] for data in recent_data], axis=0)
        
        # Adjust weights towards higher confidence modalities
        weight_keys = ['typing', 'facial', 'wearable']
        for i, key in enumerate(weight_keys):
            adjustment = (avg_confidences[i] - 0.5) * self.adaptation_rate
            self.weights[key] = max(0.1, min(0.8, self.weights[key] + adjustment))
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}

class FatigueDetectionSystem:
    """Main fatigue detection system that coordinates all components"""
    
    def __init__(self, config_file: str = "config.yaml"):
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Initialize analyzers
        self.typing_analyzer = TypingAnalyzer(
            window_size=self.config['typing']['window_size']
        ) if self.config['typing']['enabled'] else None
        
        self.facial_analyzer = FacialAnalyzer(
            camera_index=self.config['facial']['camera_index'],
            fps=self.config['facial']['fps']
        ) if self.config['facial']['enabled'] else None
        
        self.wearable_analyzer = WearableAnalyzer() if self.config['wearable']['enabled'] else None
        
        # Initialize ML models
        self.model_trainer = ModelTrainer()
        self.model_trainer.load_all_models()
        
        # Initialize fusion strategy
        self._initialize_fusion_strategy()
        
        # System state
        self.is_running = False
        self.session_start_time = None
        self.fusion_thread = None
        self.stop_event = threading.Event()
        
        # Data storage
        self.results_history = deque(maxlen=1000)
        self.database_path = self.config['storage']['database_path']
        self._initialize_database()
        
        # Alert system
        self.alert_thresholds = {
            'low': 0.3,
            'medium': 0.5,
            'high': 0.7,
            'critical': 0.9
        }
        self.last_alert_time = {}
        self.alert_cooldown = 300  # 5 minutes
        
        logging.basicConfig(level=getattr(logging, self.config['system']['log_level']))
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_file} not found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'system': {'log_level': 'INFO', 'fatigue_threshold': 0.7},
            'typing': {'enabled': True, 'window_size': 30},
            'facial': {'enabled': True, 'camera_index': 0, 'fps': 10},
            'wearable': {'enabled': True},
            'fusion': {'method': 'weighted_ensemble', 'weights': {'typing': 0.3, 'facial': 0.4, 'wearable': 0.3}},
            'storage': {'database_path': 'data/fatigue_data.db'},
            'alerts': {'enabled': True}
        }
    
    def _initialize_fusion_strategy(self):
        """Initialize the fusion strategy based on configuration"""
        fusion_method = self.config['fusion']['method']
        
        if fusion_method == 'weighted_ensemble':
            self.fusion_strategy = WeightedAverageFusion(self.config['fusion']['weights'])
        elif fusion_method == 'confidence_weighted':
            self.fusion_strategy = ConfidenceWeightedFusion()
        elif fusion_method == 'adaptive':
            self.fusion_strategy = AdaptiveFusion(self.config['fusion']['weights'])
        else:
            self.logger.warning(f"Unknown fusion method: {fusion_method}, using weighted average")
            self.fusion_strategy = WeightedAverageFusion(self.config['fusion']['weights'])
    
    def _initialize_database(self):
        """Initialize SQLite database for data storage"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS fatigue_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_start TIMESTAMP,
                    session_end TIMESTAMP,
                    duration REAL,
                    avg_fatigue_score REAL,
                    max_fatigue_score REAL,
                    alerts_count INTEGER
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS fatigue_measurements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP,
                    overall_score REAL,
                    fatigue_level TEXT,
                    typing_score REAL,
                    facial_score REAL,
                    wearable_score REAL,
                    confidence REAL,
                    alert_level TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
    
    def start_monitoring(self):
        """Start the fatigue detection system"""
        if self.is_running:
            return
        
        self.is_running = True
        self.session_start_time = datetime.now()
        self.stop_event.clear()
        
        # Start individual analyzers
        if self.typing_analyzer:
            self.typing_analyzer.start_monitoring()
        
        if self.facial_analyzer:
            self.facial_analyzer.start_monitoring()
        
        if self.wearable_analyzer:
            self.wearable_analyzer.start_monitoring()
        
        # Start fusion thread
        self.fusion_thread = threading.Thread(target=self._fusion_loop)
        self.fusion_thread.start()
        
        self.logger.info("Fatigue detection system started")
    
    def stop_monitoring(self):
        """Stop the fatigue detection system"""
        if not self.is_running:
            return
        
        self.is_running = False
        self.stop_event.set()
        
        # Stop individual analyzers
        if self.typing_analyzer:
            self.typing_analyzer.stop_monitoring()
        
        if self.facial_analyzer:
            self.facial_analyzer.stop_monitoring()
        
        if self.wearable_analyzer:
            self.wearable_analyzer.stop_monitoring()
        
        # Wait for fusion thread to finish
        if self.fusion_thread:
            self.fusion_thread.join()
        
        # Save session data
        self._save_session_data()
        
        self.logger.info("Fatigue detection system stopped")
    
    def _fusion_loop(self):
        """Main fusion loop that runs in a separate thread"""
        while not self.stop_event.is_set():
            try:
                # Collect data from all modalities
                result = self._perform_fusion()
                
                if result:
                    # Store result
                    self.results_history.append(result)
                    
                    # Save to database
                    self._save_measurement(result)
                    
                    # Check for alerts
                    self._check_alerts(result)
                
                # Wait for next iteration
                time.sleep(self.config['system']['data_collection_interval'])
                
            except Exception as e:
                self.logger.error(f"Error in fusion loop: {e}")
                time.sleep(1)
    
    def _perform_fusion(self) -> Optional[FusionResult]:
        """Perform multi-modal fusion"""
        try:
            # Collect features from each modality
            typing_features = {}
            facial_features = {}
            wearable_features = {}
            
            typing_indicators = {'overall_fatigue': 0.0}
            facial_indicators = {'overall_fatigue': 0.0}
            wearable_indicators = {'overall_fatigue': 0.0}
            
            typing_conf = 0.0
            facial_conf = 0.0
            wearable_conf = 0.0
            
            # Get typing data
            if self.typing_analyzer and self.typing_analyzer.is_monitoring:
                typing_features = self.typing_analyzer.extract_features()
                typing_indicators = self.typing_analyzer.get_fatigue_indicators()
                typing_conf = min(1.0, len(self.typing_analyzer.stats_window) / 30.0)  # Confidence based on data availability
            
            # Get facial data
            if self.facial_analyzer and self.facial_analyzer.is_monitoring:
                facial_features = self.facial_analyzer.extract_features()
                facial_indicators = self.facial_analyzer.get_fatigue_indicators()
                facial_conf = min(1.0, len(self.facial_analyzer.analysis_window) / 50.0)
            
            # Get wearable data
            if self.wearable_analyzer and self.wearable_analyzer.is_monitoring:
                wearable_features = self.wearable_analyzer.extract_features()
                wearable_indicators = self.wearable_analyzer.get_fatigue_indicators()
                wearable_conf = min(1.0, len(self.wearable_analyzer.analysis_window) / 100.0)
            
            # Check if we have any data
            if not any([typing_features, facial_features, wearable_features]):
                return None
            
            # Get individual fatigue scores
            typing_score = typing_indicators.get('overall_fatigue', 0.0)
            facial_score = facial_indicators.get('overall_fatigue', 0.0)
            wearable_score = wearable_indicators.get('overall_fatigue', 0.0)
            
            # Perform fusion
            fused_score, fused_confidence = self.fusion_strategy.fuse(
                typing_score, facial_score, wearable_score,
                typing_conf, facial_conf, wearable_conf
            )
            
            # Determine fatigue level
            fatigue_level = self._determine_fatigue_level(fused_score)
            
            # Get ML predictions if available
            ml_predictions = self._get_ml_predictions(typing_features, facial_features, wearable_features)
            
            # Determine alert level
            alert_level = self._determine_alert_level(fused_score, fatigue_level)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                fused_score, typing_score, facial_score, wearable_score, fatigue_level
            )
            
            # Create fusion result
            result = FusionResult(
                overall_fatigue_score=fused_score,
                fatigue_level=fatigue_level,
                confidence=fused_confidence,
                timestamp=datetime.now(),
                typing_score=typing_score,
                facial_score=facial_score,
                wearable_score=wearable_score,
                typing_confidence=typing_conf,
                facial_confidence=facial_conf,
                wearable_confidence=wearable_conf,
                typing_prediction=typing_indicators,
                facial_prediction=facial_indicators,
                wearable_prediction=wearable_indicators,
                ml_predictions=ml_predictions,
                alert_level=alert_level,
                recommendations=recommendations
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in fusion: {e}")
            return None
    
    def _get_ml_predictions(self, typing_features: Dict, facial_features: Dict, 
                          wearable_features: Dict) -> Optional[Dict]:
        """Get predictions from ML models"""
        try:
            # Combine all features
            all_features = {}
            all_features.update(typing_features)
            all_features.update(facial_features)
            all_features.update(wearable_features)
            
            if not all_features:
                return None
            
            # Get predictions from all models
            predictions = self.model_trainer.predict_with_all_models(all_features)
            
            # Convert to serializable format
            ml_predictions = {}
            for model_name, prediction in predictions.items():
                ml_predictions[model_name] = {
                    'fatigue_probability': prediction.fatigue_probability,
                    'fatigue_level': prediction.fatigue_level,
                    'confidence': prediction.confidence
                }
            
            return ml_predictions
            
        except Exception as e:
            self.logger.error(f"Error getting ML predictions: {e}")
            return None
    
    def _determine_fatigue_level(self, score: float) -> str:
        """Determine fatigue level from score"""
        if score < 0.2:
            return 'low'
        elif score < 0.5:
            return 'moderate'
        elif score < 0.8:
            return 'high'
        else:
            return 'critical'
    
    def _determine_alert_level(self, score: float, fatigue_level: str) -> str:
        """Determine alert level"""
        if not self.config['alerts']['enabled']:
            return 'none'
        
        if score >= self.alert_thresholds['critical']:
            return 'critical'
        elif score >= self.alert_thresholds['high']:
            return 'high'
        elif score >= self.alert_thresholds['medium']:
            return 'medium'
        elif score >= self.alert_thresholds['low']:
            return 'low'
        else:
            return 'none'
    
    def _generate_recommendations(self, overall_score: float, typing_score: float,
                                facial_score: float, wearable_score: float,
                                fatigue_level: str) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []
        
        if fatigue_level == 'critical':
            recommendations.append("🚨 Critical fatigue detected! Take an immediate break.")
            recommendations.append("💤 Consider ending work session and getting rest.")
        elif fatigue_level == 'high':
            recommendations.append("⚠️ High fatigue detected. Take a 15-30 minute break.")
            recommendations.append("🚶 Try light physical activity or stretching.")
        elif fatigue_level == 'moderate':
            recommendations.append("⚡ Moderate fatigue detected. Consider a short break.")
            recommendations.append("💧 Stay hydrated and check your posture.")
        
        # Specific recommendations based on dominant fatigue source
        max_score = max(typing_score, facial_score, wearable_score)
        
        if typing_score == max_score and typing_score > 0.6:
            recommendations.append("⌨️ Typing patterns indicate fatigue. Take breaks from typing.")
            recommendations.append("🖐️ Try hand and wrist exercises.")
        
        if facial_score == max_score and facial_score > 0.6:
            recommendations.append("👁️ Eye fatigue detected. Rest your eyes and blink more often.")
            recommendations.append("💡 Check lighting conditions and screen brightness.")
        
        if wearable_score == max_score and wearable_score > 0.6:
            recommendations.append("❤️ Physiological signs of fatigue. Focus on recovery.")
            recommendations.append("🧘 Consider relaxation techniques or deep breathing.")
        
        # General recommendations
        if overall_score > 0.4:
            recommendations.append("☕ Stay hydrated and avoid excessive caffeine.")
            recommendations.append("🌿 Ensure good ventilation and comfortable temperature.")
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    def _check_alerts(self, result: FusionResult):
        """Check and trigger alerts if necessary"""
        if result.alert_level == 'none':
            return
        
        current_time = time.time()
        
        # Check cooldown
        if (result.alert_level in self.last_alert_time and 
            current_time - self.last_alert_time[result.alert_level] < self.alert_cooldown):
            return
        
        # Trigger alert
        self._trigger_alert(result)
        self.last_alert_time[result.alert_level] = current_time
    
    def _trigger_alert(self, result: FusionResult):
        """Trigger an alert"""
        alert_message = f"Fatigue Alert: {result.fatigue_level.upper()} level detected"
        
        self.logger.warning(alert_message)
        
        # Here you could add additional alert mechanisms:
        # - Email notifications
        # - Desktop notifications
        # - Sound alerts
        # - Integration with external systems
        
        if self.config['alerts'].get('sound_alerts', False):
            # Play alert sound (implementation depends on platform)
            pass
    
    def _save_measurement(self, result: FusionResult):
        """Save measurement to database"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO fatigue_measurements 
                (timestamp, overall_score, fatigue_level, typing_score, facial_score, 
                 wearable_score, confidence, alert_level)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.timestamp,
                result.overall_fatigue_score,
                result.fatigue_level,
                result.typing_score,
                result.facial_score,
                result.wearable_score,
                result.confidence,
                result.alert_level
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error saving measurement: {e}")
    
    def _save_session_data(self):
        """Save session data to database"""
        if not self.session_start_time or not self.results_history:
            return
        
        try:
            session_end = datetime.now()
            duration = (session_end - self.session_start_time).total_seconds()
            
            scores = [r.overall_fatigue_score for r in self.results_history]
            avg_score = np.mean(scores)
            max_score = np.max(scores)
            
            alerts_count = sum(1 for r in self.results_history if r.alert_level != 'none')
            
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO fatigue_sessions 
                (session_start, session_end, duration, avg_fatigue_score, 
                 max_fatigue_score, alerts_count)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                self.session_start_time,
                session_end,
                duration,
                avg_score,
                max_score,
                alerts_count
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error saving session data: {e}")
    
    def get_current_status(self) -> SystemStatus:
        """Get current system status"""
        current_time = datetime.now()
        session_duration = 0
        
        if self.session_start_time:
            session_duration = (current_time - self.session_start_time).total_seconds()
        
        # Count total sessions
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM fatigue_sessions")
            total_sessions = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM fatigue_measurements WHERE alert_level != 'none'")
            alerts_count = cursor.fetchone()[0]
            
            conn.close()
        except:
            total_sessions = 0
            alerts_count = 0
        
        return SystemStatus(
            is_active=self.is_running,
            typing_active=self.typing_analyzer.is_monitoring if self.typing_analyzer else False,
            facial_active=self.facial_analyzer.is_monitoring if self.facial_analyzer else False,
            wearable_active=self.wearable_analyzer.is_monitoring if self.wearable_analyzer else False,
            last_update=current_time,
            total_sessions=total_sessions,
            current_session_duration=session_duration,
            alerts_count=alerts_count
        )
    
    def get_latest_result(self) -> Optional[FusionResult]:
        """Get the latest fusion result"""
        return self.results_history[-1] if self.results_history else None
    
    def get_recent_results(self, minutes: int = 30) -> List[FusionResult]:
        """Get recent results within specified time window"""
        if not self.results_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [r for r in self.results_history if r.timestamp >= cutoff_time]
    
    def get_session_summary(self) -> Dict:
        """Get summary of current session"""
        if not self.results_history:
            return {}
        
        scores = [r.overall_fatigue_score for r in self.results_history]
        typing_scores = [r.typing_score for r in self.results_history]
        facial_scores = [r.facial_score for r in self.results_history]
        wearable_scores = [r.wearable_score for r in self.results_history]
        
        alerts = [r for r in self.results_history if r.alert_level != 'none']
        
        return {
            'session_duration': (datetime.now() - self.session_start_time).total_seconds() if self.session_start_time else 0,
            'total_measurements': len(self.results_history),
            'average_fatigue_score': np.mean(scores),
            'maximum_fatigue_score': np.max(scores),
            'current_fatigue_score': scores[-1],
            'average_typing_score': np.mean(typing_scores),
            'average_facial_score': np.mean(facial_scores),
            'average_wearable_score': np.mean(wearable_scores),
            'total_alerts': len(alerts),
            'alert_breakdown': {
                'low': sum(1 for a in alerts if a.alert_level == 'low'),
                'medium': sum(1 for a in alerts if a.alert_level == 'medium'),
                'high': sum(1 for a in alerts if a.alert_level == 'high'),
                'critical': sum(1 for a in alerts if a.alert_level == 'critical')
            },
            'recommendations': self.results_history[-1].recommendations if self.results_history else []
        }

if __name__ == "__main__":
    # Example usage
    system = FatigueDetectionSystem()
    
    try:
        print("Starting fatigue detection system...")
        system.start_monitoring()
        
        # Monitor for 2 minutes
        for i in range(120):
            time.sleep(1)
            
            # Print status every 10 seconds
            if i % 10 == 0:
                latest_result = system.get_latest_result()
                if latest_result:
                    print(f"\nTime: {i}s")
                    print(f"Overall Fatigue: {latest_result.overall_fatigue_score:.3f} ({latest_result.fatigue_level})")
                    print(f"Typing: {latest_result.typing_score:.3f}, "
                          f"Facial: {latest_result.facial_score:.3f}, "
                          f"Wearable: {latest_result.wearable_score:.3f}")
                    print(f"Alert Level: {latest_result.alert_level}")
                    if latest_result.recommendations:
                        print(f"Recommendations: {latest_result.recommendations[0]}")
        
        # Print session summary
        print("\nSession Summary:")
        summary = system.get_session_summary()
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
    except KeyboardInterrupt:
        print("\nStopping system...")
    finally:
        system.stop_monitoring()