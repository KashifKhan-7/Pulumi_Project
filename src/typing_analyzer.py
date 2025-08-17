"""
Typing Pattern Analyzer for Fatigue Detection

This module captures and analyzes typing patterns to detect signs of fatigue
including keystroke dynamics, timing patterns, and typing behavior changes.
"""

import time
import threading
import numpy as np
import pandas as pd
from collections import deque, defaultdict
from pynput import keyboard
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class KeystrokeEvent:
    """Represents a single keystroke event"""
    key: str
    timestamp: float
    event_type: str  # 'press' or 'release'
    duration: Optional[float] = None  # For key releases

class TypingAnalyzer:
    """Analyzes typing patterns for fatigue detection"""
    
    def __init__(self, window_size: int = 30, buffer_size: int = 1000):
        self.window_size = window_size  # Analysis window in seconds
        self.buffer_size = buffer_size
        
        # Data storage
        self.keystroke_buffer = deque(maxlen=buffer_size)
        self.typing_sessions = []
        self.key_press_times = {}  # Track press times for duration calculation
        
        # Statistics tracking
        self.stats_window = deque(maxlen=int(window_size))
        self.current_session_start = time.time()
        
        # Keyboard listener
        self.listener = None
        self.is_monitoring = False
        
        # Feature cache
        self.feature_cache = {}
        self.last_feature_update = 0
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def start_monitoring(self):
        """Start monitoring keyboard input"""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.current_session_start = time.time()
        
        self.listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release
        )
        self.listener.start()
        self.logger.info("Typing monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring keyboard input"""
        if not self.is_monitoring:
            return
            
        self.is_monitoring = False
        if self.listener:
            self.listener.stop()
        self.logger.info("Typing monitoring stopped")
    
    def _on_key_press(self, key):
        """Handle key press events"""
        try:
            timestamp = time.time()
            key_str = self._key_to_string(key)
            
            # Store press time for duration calculation
            self.key_press_times[key_str] = timestamp
            
            # Create keystroke event
            event = KeystrokeEvent(
                key=key_str,
                timestamp=timestamp,
                event_type='press'
            )
            
            self.keystroke_buffer.append(event)
            self._update_stats_window()
            
        except Exception as e:
            self.logger.error(f"Error in key press handler: {e}")
    
    def _on_key_release(self, key):
        """Handle key release events"""
        try:
            timestamp = time.time()
            key_str = self._key_to_string(key)
            
            # Calculate key duration
            duration = None
            if key_str in self.key_press_times:
                duration = timestamp - self.key_press_times[key_str]
                del self.key_press_times[key_str]
            
            # Create keystroke event
            event = KeystrokeEvent(
                key=key_str,
                timestamp=timestamp,
                event_type='release',
                duration=duration
            )
            
            self.keystroke_buffer.append(event)
            
        except Exception as e:
            self.logger.error(f"Error in key release handler: {e}")
    
    def _key_to_string(self, key) -> str:
        """Convert key object to string representation"""
        try:
            if hasattr(key, 'char') and key.char is not None:
                return key.char
            else:
                return str(key).replace('Key.', '')
        except:
            return 'unknown'
    
    def _update_stats_window(self):
        """Update the sliding window for statistics"""
        current_time = time.time()
        
        # Remove old events from stats window
        while (self.stats_window and 
               current_time - self.stats_window[0].timestamp > self.window_size):
            self.stats_window.popleft()
        
        # Add recent events to stats window
        for event in reversed(self.keystroke_buffer):
            if current_time - event.timestamp <= self.window_size:
                if event not in self.stats_window:
                    self.stats_window.appendleft(event)
            else:
                break
    
    def extract_features(self) -> Dict[str, float]:
        """Extract typing pattern features for fatigue detection"""
        current_time = time.time()
        
        # Use cached features if recent enough (within 1 second)
        if (current_time - self.last_feature_update < 1.0 and 
            self.feature_cache):
            return self.feature_cache
        
        self._update_stats_window()
        
        if len(self.stats_window) < 10:  # Need minimum data
            return self._get_default_features()
        
        features = {}
        
        # Extract timing features
        features.update(self._extract_timing_features())
        
        # Extract rhythm features
        features.update(self._extract_rhythm_features())
        
        # Extract error and correction features
        features.update(self._extract_error_features())
        
        # Extract pressure and intensity features
        features.update(self._extract_pressure_features())
        
        # Extract behavioral features
        features.update(self._extract_behavioral_features())
        
        self.feature_cache = features
        self.last_feature_update = current_time
        
        return features
    
    def _extract_timing_features(self) -> Dict[str, float]:
        """Extract timing-related features"""
        press_events = [e for e in self.stats_window if e.event_type == 'press']
        
        if len(press_events) < 2:
            return {
                'avg_keystroke_interval': 0.0,
                'keystroke_interval_std': 0.0,
                'typing_speed_wpm': 0.0,
                'typing_speed_cps': 0.0
            }
        
        # Calculate inter-keystroke intervals
        intervals = []
        for i in range(1, len(press_events)):
            interval = press_events[i].timestamp - press_events[i-1].timestamp
            intervals.append(interval)
        
        intervals = np.array(intervals)
        
        # Calculate typing speed
        time_span = press_events[-1].timestamp - press_events[0].timestamp
        char_count = len([e for e in press_events if len(e.key) == 1])
        
        cps = char_count / max(time_span, 0.1)  # Characters per second
        wpm = (char_count / 5) / max(time_span / 60, 0.1)  # Words per minute
        
        return {
            'avg_keystroke_interval': np.mean(intervals),
            'keystroke_interval_std': np.std(intervals),
            'keystroke_interval_cv': np.std(intervals) / max(np.mean(intervals), 0.001),
            'typing_speed_wpm': wpm,
            'typing_speed_cps': cps
        }
    
    def _extract_rhythm_features(self) -> Dict[str, float]:
        """Extract typing rhythm and flow features"""
        press_events = [e for e in self.stats_window if e.event_type == 'press']
        
        if len(press_events) < 5:
            return {
                'rhythm_regularity': 0.0,
                'flow_interruptions': 0.0,
                'pause_frequency': 0.0,
                'long_pause_ratio': 0.0
            }
        
        intervals = []
        for i in range(1, len(press_events)):
            interval = press_events[i].timestamp - press_events[i-1].timestamp
            intervals.append(interval)
        
        intervals = np.array(intervals)
        
        # Rhythm regularity (inverse of coefficient of variation)
        rhythm_regularity = 1.0 / (1.0 + np.std(intervals) / max(np.mean(intervals), 0.001))
        
        # Flow interruptions (pauses > 2 seconds)
        long_pauses = np.sum(intervals > 2.0)
        long_pause_ratio = long_pauses / len(intervals)
        
        # Pause frequency (pauses > 0.5 seconds)
        pauses = np.sum(intervals > 0.5)
        pause_frequency = pauses / len(intervals)
        
        # Flow interruptions per minute
        time_span = max(press_events[-1].timestamp - press_events[0].timestamp, 0.1)
        flow_interruptions = (long_pauses / time_span) * 60
        
        return {
            'rhythm_regularity': rhythm_regularity,
            'flow_interruptions': flow_interruptions,
            'pause_frequency': pause_frequency,
            'long_pause_ratio': long_pause_ratio
        }
    
    def _extract_error_features(self) -> Dict[str, float]:
        """Extract error and correction pattern features"""
        press_events = [e for e in self.stats_window if e.event_type == 'press']
        
        # Count backspace/delete usage (error corrections)
        correction_keys = ['backspace', 'delete', 'BackSpace', 'Delete']
        corrections = len([e for e in press_events if e.key in correction_keys])
        
        total_keys = len(press_events)
        error_rate = corrections / max(total_keys, 1)
        
        # Correction bursts (multiple corrections in sequence)
        correction_bursts = 0
        in_burst = False
        
        for event in press_events:
            if event.key in correction_keys:
                if not in_burst:
                    correction_bursts += 1
                    in_burst = True
            else:
                in_burst = False
        
        return {
            'error_rate': error_rate,
            'correction_frequency': corrections / max(len(press_events) / 60, 0.1),
            'correction_bursts': correction_bursts
        }
    
    def _extract_pressure_features(self) -> Dict[str, float]:
        """Extract key pressure and intensity features"""
        release_events = [e for e in self.stats_window 
                         if e.event_type == 'release' and e.duration is not None]
        
        if not release_events:
            return {
                'avg_key_duration': 0.0,
                'key_duration_std': 0.0,
                'pressure_variance': 0.0
            }
        
        durations = [e.duration for e in release_events]
        durations = np.array(durations)
        
        # Filter out outliers (very long key holds)
        durations = durations[durations < 2.0]
        
        if len(durations) == 0:
            return {
                'avg_key_duration': 0.0,
                'key_duration_std': 0.0,
                'pressure_variance': 0.0
            }
        
        return {
            'avg_key_duration': np.mean(durations),
            'key_duration_std': np.std(durations),
            'pressure_variance': np.var(durations)
        }
    
    def _extract_behavioral_features(self) -> Dict[str, float]:
        """Extract behavioral pattern features"""
        press_events = [e for e in self.stats_window if e.event_type == 'press']
        
        if not press_events:
            return {
                'activity_level': 0.0,
                'typing_consistency': 0.0,
                'micro_break_frequency': 0.0
            }
        
        # Activity level (keystrokes per minute)
        time_span = max(press_events[-1].timestamp - press_events[0].timestamp, 0.1)
        activity_level = (len(press_events) / time_span) * 60
        
        # Typing consistency (regularity of typing bursts)
        # Divide into 5-second chunks and measure consistency
        chunk_size = 5.0
        chunks = defaultdict(int)
        start_time = press_events[0].timestamp
        
        for event in press_events:
            chunk_idx = int((event.timestamp - start_time) / chunk_size)
            chunks[chunk_idx] += 1
        
        chunk_counts = list(chunks.values())
        typing_consistency = 1.0 / (1.0 + np.std(chunk_counts)) if chunk_counts else 0.0
        
        # Micro-break frequency (gaps of 3-10 seconds)
        intervals = []
        for i in range(1, len(press_events)):
            interval = press_events[i].timestamp - press_events[i-1].timestamp
            intervals.append(interval)
        
        micro_breaks = len([i for i in intervals if 3.0 <= i <= 10.0])
        micro_break_frequency = micro_breaks / max(len(intervals), 1)
        
        return {
            'activity_level': activity_level,
            'typing_consistency': typing_consistency,
            'micro_break_frequency': micro_break_frequency
        }
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default feature values when insufficient data"""
        return {
            'avg_keystroke_interval': 0.0,
            'keystroke_interval_std': 0.0,
            'keystroke_interval_cv': 0.0,
            'typing_speed_wpm': 0.0,
            'typing_speed_cps': 0.0,
            'rhythm_regularity': 0.0,
            'flow_interruptions': 0.0,
            'pause_frequency': 0.0,
            'long_pause_ratio': 0.0,
            'error_rate': 0.0,
            'correction_frequency': 0.0,
            'correction_bursts': 0.0,
            'avg_key_duration': 0.0,
            'key_duration_std': 0.0,
            'pressure_variance': 0.0,
            'activity_level': 0.0,
            'typing_consistency': 0.0,
            'micro_break_frequency': 0.0
        }
    
    def get_fatigue_indicators(self) -> Dict[str, float]:
        """Get fatigue-specific indicators from typing patterns"""
        features = self.extract_features()
        
        indicators = {}
        
        # Slower typing speed indicates fatigue
        indicators['speed_fatigue'] = max(0, 1.0 - features['typing_speed_wpm'] / 60.0)
        
        # Irregular rhythm indicates fatigue
        indicators['rhythm_fatigue'] = 1.0 - features['rhythm_regularity']
        
        # Increased errors indicate fatigue
        indicators['error_fatigue'] = min(1.0, features['error_rate'] * 10)
        
        # Longer pauses indicate fatigue
        indicators['pause_fatigue'] = min(1.0, features['pause_frequency'] * 2)
        
        # Inconsistent key pressure indicates fatigue
        indicators['pressure_fatigue'] = min(1.0, features['pressure_variance'] * 100)
        
        # Overall fatigue score (weighted combination)
        weights = {
            'speed_fatigue': 0.25,
            'rhythm_fatigue': 0.25,
            'error_fatigue': 0.20,
            'pause_fatigue': 0.15,
            'pressure_fatigue': 0.15
        }
        
        overall_fatigue = sum(indicators[key] * weights[key] for key in weights)
        indicators['overall_fatigue'] = overall_fatigue
        
        return indicators
    
    def get_session_summary(self) -> Dict:
        """Get summary of current typing session"""
        if not self.keystroke_buffer:
            return {}
        
        session_duration = time.time() - self.current_session_start
        total_keystrokes = len(self.keystroke_buffer)
        
        features = self.extract_features()
        indicators = self.get_fatigue_indicators()
        
        return {
            'session_duration': session_duration,
            'total_keystrokes': total_keystrokes,
            'current_features': features,
            'fatigue_indicators': indicators,
            'timestamp': datetime.now().isoformat()
        }
    
    def reset_session(self):
        """Reset current typing session"""
        self.keystroke_buffer.clear()
        self.stats_window.clear()
        self.key_press_times.clear()
        self.feature_cache.clear()
        self.current_session_start = time.time()
        self.logger.info("Typing session reset")

if __name__ == "__main__":
    # Example usage
    analyzer = TypingAnalyzer()
    
    try:
        analyzer.start_monitoring()
        print("Typing analysis started. Type something...")
        
        # Monitor for 60 seconds
        time.sleep(60)
        
        # Get results
        features = analyzer.extract_features()
        indicators = analyzer.get_fatigue_indicators()
        summary = analyzer.get_session_summary()
        
        print("\nTyping Features:")
        for key, value in features.items():
            print(f"  {key}: {value:.3f}")
        
        print("\nFatigue Indicators:")
        for key, value in indicators.items():
            print(f"  {key}: {value:.3f}")
        
        print(f"\nOverall Fatigue Score: {indicators['overall_fatigue']:.3f}")
        
    except KeyboardInterrupt:
        print("\nStopping analysis...")
    finally:
        analyzer.stop_monitoring()