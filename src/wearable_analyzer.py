"""
Wearable Data Analyzer for Fatigue Detection

This module integrates with wearable devices to collect and analyze
physiological data including heart rate variability, activity levels,
sleep quality, and stress indicators for fatigue detection.
"""

import time
import threading
import numpy as np
import pandas as pd
import requests
import json
import logging
from collections import deque
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy import signal
from scipy.stats import entropy
import psutil

@dataclass
class WearableData:
    """Represents wearable sensor data"""
    timestamp: float
    heart_rate: Optional[float]
    heart_rate_variability: Optional[float]
    activity_level: Optional[float]
    steps: Optional[int]
    calories_burned: Optional[float]
    sleep_score: Optional[float]
    stress_level: Optional[float]
    skin_temperature: Optional[float]
    blood_oxygen: Optional[float]

@dataclass
class DeviceConfig:
    """Configuration for a wearable device"""
    device_type: str
    device_id: str
    api_endpoint: Optional[str]
    bluetooth_address: Optional[str]
    polling_interval: int  # seconds
    enabled: bool

class WearableAnalyzer:
    """Analyzes wearable data for fatigue detection"""
    
    def __init__(self, config_file: Optional[str] = None):
        # Device configurations
        self.devices = []
        self.device_connections = {}
        
        # Data storage
        self.data_buffer = deque(maxlen=3600)  # Store 1 hour of data (assuming 1-second intervals)
        self.analysis_window = deque(maxlen=1800)  # 30-minute analysis window
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_threads = {}
        self.stop_event = threading.Event()
        
        # Feature cache
        self.feature_cache = {}
        self.last_feature_update = 0
        
        # HRV analysis parameters
        self.hrv_window_size = 300  # 5 minutes for HRV analysis
        self.activity_threshold = 50  # Steps per minute threshold for activity
        
        # Simulated data for demo purposes
        self.simulate_data = True
        self.simulation_thread = None
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize default devices
        self._initialize_default_devices()
    
    def _initialize_default_devices(self):
        """Initialize default device configurations"""
        default_devices = [
            DeviceConfig(
                device_type="heart_rate_monitor",
                device_id="hrm_001",
                api_endpoint=None,
                bluetooth_address=None,
                polling_interval=1,
                enabled=True
            ),
            DeviceConfig(
                device_type="fitness_tracker",
                device_id="fit_001",
                api_endpoint=None,
                bluetooth_address=None,
                polling_interval=5,
                enabled=True
            ),
            DeviceConfig(
                device_type="smartwatch",
                device_id="watch_001",
                api_endpoint=None,
                bluetooth_address=None,
                polling_interval=10,
                enabled=True
            )
        ]
        
        self.devices.extend(default_devices)
    
    def add_device(self, device_config: DeviceConfig):
        """Add a new wearable device"""
        self.devices.append(device_config)
        self.logger.info(f"Added device: {device_config.device_type} - {device_config.device_id}")
    
    def start_monitoring(self):
        """Start monitoring all enabled wearable devices"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.stop_event.clear()
        
        # Start monitoring threads for each enabled device
        for device in self.devices:
            if device.enabled:
                thread = threading.Thread(
                    target=self._monitor_device,
                    args=(device,),
                    name=f"monitor_{device.device_id}"
                )
                thread.start()
                self.monitoring_threads[device.device_id] = thread
        
        # Start simulation thread if no real devices
        if self.simulate_data:
            self.simulation_thread = threading.Thread(target=self._simulate_data)
            self.simulation_thread.start()
        
        self.logger.info("Wearable monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring all devices"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        self.stop_event.set()
        
        # Wait for all threads to finish
        for thread in self.monitoring_threads.values():
            thread.join()
        
        if self.simulation_thread:
            self.simulation_thread.join()
        
        # Close device connections
        for connection in self.device_connections.values():
            if hasattr(connection, 'close'):
                connection.close()
        
        self.monitoring_threads.clear()
        self.device_connections.clear()
        
        self.logger.info("Wearable monitoring stopped")
    
    def _monitor_device(self, device: DeviceConfig):
        """Monitor a specific device"""
        try:
            # Attempt to connect to device
            if device.bluetooth_address:
                connection = self._connect_bluetooth_device(device)
            elif device.api_endpoint:
                connection = self._connect_api_device(device)
            else:
                connection = None
            
            self.device_connections[device.device_id] = connection
            
            while not self.stop_event.is_set():
                try:
                    # Collect data from device
                    data = self._collect_device_data(device, connection)
                    
                    if data:
                        self.data_buffer.append(data)
                        self._update_analysis_window()
                    
                    time.sleep(device.polling_interval)
                    
                except Exception as e:
                    self.logger.error(f"Error collecting data from {device.device_id}: {e}")
                    time.sleep(device.polling_interval)
                    
        except Exception as e:
            self.logger.error(f"Error monitoring device {device.device_id}: {e}")
    
    def _connect_bluetooth_device(self, device: DeviceConfig):
        """Connect to a Bluetooth wearable device"""
        # Placeholder for Bluetooth connection
        # In a real implementation, this would use bluetooth libraries
        self.logger.info(f"Connecting to Bluetooth device: {device.bluetooth_address}")
        return None
    
    def _connect_api_device(self, device: DeviceConfig):
        """Connect to a device via API"""
        # Placeholder for API connection
        self.logger.info(f"Connecting to API device: {device.api_endpoint}")
        return None
    
    def _collect_device_data(self, device: DeviceConfig, connection) -> Optional[WearableData]:
        """Collect data from a specific device"""
        try:
            # This is a placeholder - real implementation would depend on device APIs
            if device.device_type == "heart_rate_monitor":
                return self._collect_heart_rate_data(device, connection)
            elif device.device_type == "fitness_tracker":
                return self._collect_fitness_data(device, connection)
            elif device.device_type == "smartwatch":
                return self._collect_smartwatch_data(device, connection)
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error collecting data from {device.device_type}: {e}")
            return None
    
    def _collect_heart_rate_data(self, device: DeviceConfig, connection) -> Optional[WearableData]:
        """Collect heart rate data"""
        # Placeholder - would interface with actual heart rate monitor
        return None
    
    def _collect_fitness_data(self, device: DeviceConfig, connection) -> Optional[WearableData]:
        """Collect fitness tracker data"""
        # Placeholder - would interface with actual fitness tracker
        return None
    
    def _collect_smartwatch_data(self, device: DeviceConfig, connection) -> Optional[WearableData]:
        """Collect smartwatch data"""
        # Placeholder - would interface with actual smartwatch
        return None
    
    def _simulate_data(self):
        """Simulate wearable data for demonstration purposes"""
        base_hr = 70  # Base heart rate
        base_activity = 30  # Base activity level
        fatigue_factor = 0.0
        
        while not self.stop_event.is_set():
            try:
                current_time = time.time()
                
                # Simulate gradual fatigue increase
                fatigue_factor = min(1.0, fatigue_factor + 0.001)
                
                # Add some randomness and circadian rhythm effects
                time_of_day = (current_time % 86400) / 86400  # 0-1 for 24 hours
                circadian_effect = 0.1 * np.sin(2 * np.pi * time_of_day - np.pi/2)  # Peak at midday
                
                # Simulate heart rate with fatigue effects
                hr_noise = np.random.normal(0, 5)
                heart_rate = base_hr + (fatigue_factor * 15) + circadian_effect * 10 + hr_noise
                heart_rate = max(50, min(120, heart_rate))  # Clamp to realistic range
                
                # Simulate HRV (decreases with fatigue)
                base_hrv = 50
                hrv = base_hrv * (1 - fatigue_factor * 0.5) + np.random.normal(0, 5)
                hrv = max(10, min(100, hrv))
                
                # Simulate activity level (decreases with fatigue)
                activity_noise = np.random.normal(0, 10)
                activity = base_activity * (1 - fatigue_factor * 0.3) + activity_noise
                activity = max(0, activity)
                
                # Simulate steps (correlated with activity)
                steps_per_minute = max(0, int(activity / 2 + np.random.normal(0, 2)))
                
                # Simulate sleep score (decreases with fatigue)
                sleep_score = 85 * (1 - fatigue_factor * 0.2) + np.random.normal(0, 5)
                sleep_score = max(0, min(100, sleep_score))
                
                # Simulate stress level (increases with fatigue)
                stress_level = fatigue_factor * 70 + np.random.normal(0, 10)
                stress_level = max(0, min(100, stress_level))
                
                # Simulate skin temperature
                base_temp = 36.5
                temp = base_temp + fatigue_factor * 0.5 + np.random.normal(0, 0.2)
                
                # Simulate blood oxygen (decreases slightly with fatigue)
                base_spo2 = 98
                spo2 = base_spo2 - fatigue_factor * 2 + np.random.normal(0, 0.5)
                spo2 = max(90, min(100, spo2))
                
                # Create wearable data point
                data = WearableData(
                    timestamp=current_time,
                    heart_rate=heart_rate,
                    heart_rate_variability=hrv,
                    activity_level=activity,
                    steps=steps_per_minute,
                    calories_burned=activity * 0.1,
                    sleep_score=sleep_score,
                    stress_level=stress_level,
                    skin_temperature=temp,
                    blood_oxygen=spo2
                )
                
                self.data_buffer.append(data)
                self._update_analysis_window()
                
                time.sleep(1)  # Generate data every second
                
            except Exception as e:
                self.logger.error(f"Error in data simulation: {e}")
                time.sleep(1)
    
    def _update_analysis_window(self):
        """Update the analysis window with recent data"""
        current_time = time.time()
        
        # Remove old data from analysis window
        while (self.analysis_window and 
               current_time - self.analysis_window[0].timestamp > 1800):  # 30 minutes
            self.analysis_window.popleft()
        
        # Add recent data to analysis window
        for data in reversed(self.data_buffer):
            if current_time - data.timestamp <= 1800:
                if data not in self.analysis_window:
                    self.analysis_window.appendleft(data)
            else:
                break
    
    def extract_features(self) -> Dict[str, float]:
        """Extract wearable features for fatigue detection"""
        current_time = time.time()
        
        # Use cached features if recent enough
        if (current_time - self.last_feature_update < 5.0 and self.feature_cache):
            return self.feature_cache
        
        self._update_analysis_window()
        
        if len(self.analysis_window) < 10:
            return self._get_default_features()
        
        features = {}
        
        # Extract heart rate features
        features.update(self._extract_heart_rate_features())
        
        # Extract activity features
        features.update(self._extract_activity_features())
        
        # Extract sleep and recovery features
        features.update(self._extract_sleep_features())
        
        # Extract stress indicators
        features.update(self._extract_stress_features())
        
        # Extract physiological features
        features.update(self._extract_physiological_features())
        
        self.feature_cache = features
        self.last_feature_update = current_time
        
        return features
    
    def _extract_heart_rate_features(self) -> Dict[str, float]:
        """Extract heart rate and HRV features"""
        hr_data = [d.heart_rate for d in self.analysis_window if d.heart_rate is not None]
        hrv_data = [d.heart_rate_variability for d in self.analysis_window if d.heart_rate_variability is not None]
        
        if not hr_data:
            return {
                'avg_heart_rate': 70.0,
                'heart_rate_std': 0.0,
                'heart_rate_trend': 0.0,
                'avg_hrv': 50.0,
                'hrv_std': 0.0,
                'hrv_trend': 0.0,
                'resting_heart_rate': 70.0,
                'heart_rate_recovery': 1.0
            }
        
        hr_array = np.array(hr_data)
        
        # Basic heart rate statistics
        avg_hr = np.mean(hr_array)
        hr_std = np.std(hr_array)
        
        # Heart rate trend (slope over time)
        if len(hr_data) >= 10:
            x = np.arange(len(hr_data))
            hr_trend = np.polyfit(x, hr_data, 1)[0]
        else:
            hr_trend = 0.0
        
        # Resting heart rate (10th percentile)
        resting_hr = np.percentile(hr_array, 10)
        
        # Heart rate recovery (ability to return to baseline)
        hr_recovery = self._calculate_hr_recovery(hr_data)
        
        # HRV features
        if hrv_data:
            hrv_array = np.array(hrv_data)
            avg_hrv = np.mean(hrv_array)
            hrv_std = np.std(hrv_array)
            
            if len(hrv_data) >= 10:
                x = np.arange(len(hrv_data))
                hrv_trend = np.polyfit(x, hrv_data, 1)[0]
            else:
                hrv_trend = 0.0
        else:
            avg_hrv = 50.0
            hrv_std = 0.0
            hrv_trend = 0.0
        
        return {
            'avg_heart_rate': avg_hr,
            'heart_rate_std': hr_std,
            'heart_rate_trend': hr_trend,
            'avg_hrv': avg_hrv,
            'hrv_std': hrv_std,
            'hrv_trend': hrv_trend,
            'resting_heart_rate': resting_hr,
            'heart_rate_recovery': hr_recovery
        }
    
    def _extract_activity_features(self) -> Dict[str, float]:
        """Extract activity and movement features"""
        activity_data = [d.activity_level for d in self.analysis_window if d.activity_level is not None]
        steps_data = [d.steps for d in self.analysis_window if d.steps is not None]
        calories_data = [d.calories_burned for d in self.analysis_window if d.calories_burned is not None]
        
        if not activity_data:
            return {
                'avg_activity_level': 30.0,
                'activity_std': 0.0,
                'activity_trend': 0.0,
                'total_steps': 0,
                'steps_per_minute': 0.0,
                'total_calories': 0.0,
                'sedentary_time_ratio': 0.0,
                'activity_consistency': 1.0
            }
        
        activity_array = np.array(activity_data)
        
        # Basic activity statistics
        avg_activity = np.mean(activity_array)
        activity_std = np.std(activity_array)
        
        # Activity trend
        if len(activity_data) >= 10:
            x = np.arange(len(activity_data))
            activity_trend = np.polyfit(x, activity_data, 1)[0]
        else:
            activity_trend = 0.0
        
        # Steps analysis
        if steps_data:
            total_steps = sum(steps_data)
            steps_per_minute = np.mean(steps_data)
        else:
            total_steps = 0
            steps_per_minute = 0.0
        
        # Calories
        total_calories = sum(calories_data) if calories_data else 0.0
        
        # Sedentary time (activity below threshold)
        sedentary_minutes = sum(1 for a in activity_data if a < self.activity_threshold)
        sedentary_ratio = sedentary_minutes / len(activity_data)
        
        # Activity consistency (inverse of coefficient of variation)
        activity_consistency = 1.0 / (1.0 + (activity_std / max(avg_activity, 1.0)))
        
        return {
            'avg_activity_level': avg_activity,
            'activity_std': activity_std,
            'activity_trend': activity_trend,
            'total_steps': total_steps,
            'steps_per_minute': steps_per_minute,
            'total_calories': total_calories,
            'sedentary_time_ratio': sedentary_ratio,
            'activity_consistency': activity_consistency
        }
    
    def _extract_sleep_features(self) -> Dict[str, float]:
        """Extract sleep and recovery features"""
        sleep_data = [d.sleep_score for d in self.analysis_window if d.sleep_score is not None]
        
        if not sleep_data:
            return {
                'avg_sleep_score': 85.0,
                'sleep_score_std': 0.0,
                'sleep_trend': 0.0,
                'sleep_consistency': 1.0,
                'recovery_score': 1.0
            }
        
        sleep_array = np.array(sleep_data)
        
        # Basic sleep statistics
        avg_sleep = np.mean(sleep_array)
        sleep_std = np.std(sleep_array)
        
        # Sleep trend
        if len(sleep_data) >= 5:
            x = np.arange(len(sleep_data))
            sleep_trend = np.polyfit(x, sleep_data, 1)[0]
        else:
            sleep_trend = 0.0
        
        # Sleep consistency
        sleep_consistency = 1.0 / (1.0 + (sleep_std / max(avg_sleep, 1.0)))
        
        # Recovery score (based on sleep and HRV)
        recovery_score = min(1.0, avg_sleep / 85.0)  # Normalize to 0-1
        
        return {
            'avg_sleep_score': avg_sleep,
            'sleep_score_std': sleep_std,
            'sleep_trend': sleep_trend,
            'sleep_consistency': sleep_consistency,
            'recovery_score': recovery_score
        }
    
    def _extract_stress_features(self) -> Dict[str, float]:
        """Extract stress and mental state features"""
        stress_data = [d.stress_level for d in self.analysis_window if d.stress_level is not None]
        
        if not stress_data:
            return {
                'avg_stress_level': 30.0,
                'stress_std': 0.0,
                'stress_trend': 0.0,
                'high_stress_ratio': 0.0,
                'stress_variability': 0.0
            }
        
        stress_array = np.array(stress_data)
        
        # Basic stress statistics
        avg_stress = np.mean(stress_array)
        stress_std = np.std(stress_array)
        
        # Stress trend
        if len(stress_data) >= 10:
            x = np.arange(len(stress_data))
            stress_trend = np.polyfit(x, stress_data, 1)[0]
        else:
            stress_trend = 0.0
        
        # High stress periods (stress > 70)
        high_stress_periods = sum(1 for s in stress_data if s > 70)
        high_stress_ratio = high_stress_periods / len(stress_data)
        
        # Stress variability (coefficient of variation)
        stress_variability = stress_std / max(avg_stress, 1.0)
        
        return {
            'avg_stress_level': avg_stress,
            'stress_std': stress_std,
            'stress_trend': stress_trend,
            'high_stress_ratio': high_stress_ratio,
            'stress_variability': stress_variability
        }
    
    def _extract_physiological_features(self) -> Dict[str, float]:
        """Extract physiological indicators"""
        temp_data = [d.skin_temperature for d in self.analysis_window if d.skin_temperature is not None]
        spo2_data = [d.blood_oxygen for d in self.analysis_window if d.blood_oxygen is not None]
        
        features = {}
        
        # Temperature features
        if temp_data:
            temp_array = np.array(temp_data)
            features.update({
                'avg_skin_temperature': np.mean(temp_array),
                'temperature_std': np.std(temp_array),
                'temperature_trend': np.polyfit(np.arange(len(temp_data)), temp_data, 1)[0] if len(temp_data) >= 5 else 0.0
            })
        else:
            features.update({
                'avg_skin_temperature': 36.5,
                'temperature_std': 0.0,
                'temperature_trend': 0.0
            })
        
        # Blood oxygen features
        if spo2_data:
            spo2_array = np.array(spo2_data)
            features.update({
                'avg_blood_oxygen': np.mean(spo2_array),
                'blood_oxygen_std': np.std(spo2_array),
                'low_oxygen_ratio': sum(1 for spo2 in spo2_data if spo2 < 95) / len(spo2_data)
            })
        else:
            features.update({
                'avg_blood_oxygen': 98.0,
                'blood_oxygen_std': 0.0,
                'low_oxygen_ratio': 0.0
            })
        
        return features
    
    def _calculate_hr_recovery(self, hr_data: List[float]) -> float:
        """Calculate heart rate recovery score"""
        if len(hr_data) < 60:  # Need at least 1 minute of data
            return 1.0
        
        # Find periods of elevated heart rate followed by recovery
        hr_array = np.array(hr_data)
        baseline = np.percentile(hr_array, 25)  # 25th percentile as baseline
        
        recovery_scores = []
        i = 0
        while i < len(hr_array) - 30:  # Need 30 seconds for recovery analysis
            if hr_array[i] > baseline + 10:  # Elevated heart rate
                # Look for recovery in next 30 seconds
                recovery_period = hr_array[i:i+30]
                if len(recovery_period) >= 30:
                    initial_hr = recovery_period[0]
                    final_hr = recovery_period[-1]
                    recovery_rate = (initial_hr - final_hr) / 30  # HR decrease per second
                    recovery_scores.append(max(0, recovery_rate))
                i += 30
            else:
                i += 1
        
        if recovery_scores:
            return min(1.0, np.mean(recovery_scores) * 10)  # Normalize
        else:
            return 1.0
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default feature values when insufficient data"""
        return {
            'avg_heart_rate': 70.0,
            'heart_rate_std': 0.0,
            'heart_rate_trend': 0.0,
            'avg_hrv': 50.0,
            'hrv_std': 0.0,
            'hrv_trend': 0.0,
            'resting_heart_rate': 70.0,
            'heart_rate_recovery': 1.0,
            'avg_activity_level': 30.0,
            'activity_std': 0.0,
            'activity_trend': 0.0,
            'total_steps': 0,
            'steps_per_minute': 0.0,
            'total_calories': 0.0,
            'sedentary_time_ratio': 0.0,
            'activity_consistency': 1.0,
            'avg_sleep_score': 85.0,
            'sleep_score_std': 0.0,
            'sleep_trend': 0.0,
            'sleep_consistency': 1.0,
            'recovery_score': 1.0,
            'avg_stress_level': 30.0,
            'stress_std': 0.0,
            'stress_trend': 0.0,
            'high_stress_ratio': 0.0,
            'stress_variability': 0.0,
            'avg_skin_temperature': 36.5,
            'temperature_std': 0.0,
            'temperature_trend': 0.0,
            'avg_blood_oxygen': 98.0,
            'blood_oxygen_std': 0.0,
            'low_oxygen_ratio': 0.0
        }
    
    def get_fatigue_indicators(self) -> Dict[str, float]:
        """Get fatigue-specific indicators from wearable data"""
        features = self.extract_features()
        
        indicators = {}
        
        # Heart rate fatigue (elevated resting HR, poor recovery)
        hr_fatigue = min(1.0, max(0, (features['resting_heart_rate'] - 60) / 30))
        hr_fatigue += (1.0 - features['heart_rate_recovery'])
        indicators['heart_rate_fatigue'] = min(1.0, hr_fatigue / 2)
        
        # HRV fatigue (decreased HRV indicates fatigue)
        hrv_fatigue = max(0, 1.0 - (features['avg_hrv'] / 50.0))
        indicators['hrv_fatigue'] = hrv_fatigue
        
        # Activity fatigue (decreased activity, increased sedentary time)
        activity_fatigue = features['sedentary_time_ratio']
        activity_fatigue += max(0, 1.0 - (features['avg_activity_level'] / 50.0))
        indicators['activity_fatigue'] = min(1.0, activity_fatigue / 2)
        
        # Sleep fatigue (poor sleep quality)
        sleep_fatigue = max(0, 1.0 - (features['avg_sleep_score'] / 85.0))
        indicators['sleep_fatigue'] = sleep_fatigue
        
        # Stress fatigue
        stress_fatigue = min(1.0, features['avg_stress_level'] / 100.0)
        indicators['stress_fatigue'] = stress_fatigue
        
        # Recovery fatigue (poor overall recovery)
        recovery_fatigue = 1.0 - features['recovery_score']
        indicators['recovery_fatigue'] = recovery_fatigue
        
        # Overall wearable fatigue (weighted combination)
        weights = {
            'heart_rate_fatigue': 0.20,
            'hrv_fatigue': 0.25,
            'activity_fatigue': 0.15,
            'sleep_fatigue': 0.20,
            'stress_fatigue': 0.10,
            'recovery_fatigue': 0.10
        }
        
        overall_fatigue = sum(indicators[key] * weights[key] for key in weights)
        indicators['overall_fatigue'] = overall_fatigue
        
        return indicators
    
    def get_session_summary(self) -> Dict:
        """Get summary of current wearable monitoring session"""
        if not self.analysis_window:
            return {}
        
        features = self.extract_features()
        indicators = self.get_fatigue_indicators()
        
        session_duration = time.time() - self.analysis_window[0].timestamp if self.analysis_window else 0
        
        return {
            'session_duration': session_duration,
            'total_data_points': len(self.analysis_window),
            'devices_active': len([d for d in self.devices if d.enabled]),
            'current_features': features,
            'fatigue_indicators': indicators,
            'timestamp': datetime.now().isoformat()
        }

if __name__ == "__main__":
    # Example usage
    analyzer = WearableAnalyzer()
    
    try:
        analyzer.start_monitoring()
        print("Wearable monitoring started...")
        
        # Monitor for 60 seconds
        time.sleep(60)
        
        # Get results
        features = analyzer.extract_features()
        indicators = analyzer.get_fatigue_indicators()
        summary = analyzer.get_session_summary()
        
        print("\nWearable Features:")
        for key, value in features.items():
            print(f"  {key}: {value:.3f}")
        
        print("\nFatigue Indicators:")
        for key, value in indicators.items():
            print(f"  {key}: {value:.3f}")
        
        print(f"\nOverall Wearable Fatigue Score: {indicators['overall_fatigue']:.3f}")
        
    except KeyboardInterrupt:
        print("\nStopping monitoring...")
    finally:
        analyzer.stop_monitoring()