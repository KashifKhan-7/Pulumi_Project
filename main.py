#!/usr/bin/env python3
"""
Main entry point for the Fatigue Detection System

This script provides different modes of operation:
1. Web mode - Run the web interface and API server
2. CLI mode - Run from command line with text output
3. Train mode - Train machine learning models
4. Test mode - Run system tests
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def setup_logging(level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('fatigue_detection.log')
        ]
    )

def run_web_mode(host='0.0.0.0', port=5000, debug=False):
    """Run the web interface and API server"""
    print("Starting Fatigue Detection System - Web Mode")
    print(f"Server will be available at: http://{host}:{port}")
    print("Press Ctrl+C to stop the server")
    
    try:
        from app import app, socketio, initialize_system
        
        # Initialize system
        success = initialize_system()
        if not success:
            print("ERROR: Failed to initialize system")
            return 1
        
        # Run the Flask app with SocketIO
        socketio.run(app, 
                    host=host, 
                    port=port, 
                    debug=debug,
                    allow_unsafe_werkzeug=True)
        
        return 0
        
    except KeyboardInterrupt:
        print("\nShutting down server...")
        return 0
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

def run_cli_mode(duration=60):
    """Run the system in CLI mode with text output"""
    print("Starting Fatigue Detection System - CLI Mode")
    print(f"Monitoring for {duration} seconds...")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        from fusion_system import FatigueDetectionSystem
        import time
        
        # Initialize system
        system = FatigueDetectionSystem()
        
        # Start monitoring
        system.start_monitoring()
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                time.sleep(5)  # Update every 5 seconds
                
                # Get latest result
                result = system.get_latest_result()
                
                if result:
                    print(f"\n--- {result.timestamp.strftime('%H:%M:%S')} ---")
                    print(f"Overall Fatigue: {result.overall_fatigue_score:.3f} ({result.fatigue_level})")
                    print(f"Typing: {result.typing_score:.3f} | "
                          f"Facial: {result.facial_score:.3f} | "
                          f"Wearable: {result.wearable_score:.3f}")
                    print(f"Confidence: {result.confidence:.3f}")
                    
                    if result.alert_level != 'none':
                        print(f"🚨 ALERT: {result.alert_level.upper()}")
                    
                    if result.recommendations:
                        print(f"💡 {result.recommendations[0]}")
                else:
                    print(".", end="", flush=True)
        
        except KeyboardInterrupt:
            print("\n\nStopping monitoring...")
        
        # Stop system and show summary
        system.stop_monitoring()
        
        # Print session summary
        summary = system.get_session_summary()
        if summary:
            print("\n" + "="*50)
            print("SESSION SUMMARY")
            print("="*50)
            print(f"Duration: {summary['session_duration']:.1f} seconds")
            print(f"Data Points: {summary['total_measurements']}")
            print(f"Average Fatigue: {summary['average_fatigue_score']:.3f}")
            print(f"Peak Fatigue: {summary['maximum_fatigue_score']:.3f}")
            print(f"Total Alerts: {summary['total_alerts']}")
        
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

def run_train_mode():
    """Train machine learning models"""
    print("Starting Model Training...")
    
    try:
        from ai_models import ModelTrainer
        
        trainer = ModelTrainer()
        
        print("Training models with synthetic data...")
        results = trainer.train_all_models()
        
        print("\nTraining Results:")
        print("="*50)
        
        for model_name, performance in results.items():
            if performance:
                print(f"\n{model_name.upper()}:")
                print(f"  Accuracy: {performance.accuracy:.3f}")
                print(f"  Precision: {performance.precision:.3f}")
                print(f"  Recall: {performance.recall:.3f}")
                print(f"  F1 Score: {performance.f1_score:.3f}")
                print(f"  AUC Score: {performance.auc_score:.3f}")
            else:
                print(f"\n{model_name.upper()}: FAILED")
        
        print(f"\nModels saved to: models/")
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

def run_test_mode():
    """Run system tests"""
    print("Running System Tests...")
    
    try:
        # Test individual modules
        test_results = {}
        
        # Test typing analyzer
        print("\nTesting Typing Analyzer...")
        try:
            from typing_analyzer import TypingAnalyzer
            analyzer = TypingAnalyzer()
            features = analyzer.extract_features()
            test_results['typing'] = len(features) > 0
            print("✓ Typing Analyzer: PASSED")
        except Exception as e:
            test_results['typing'] = False
            print(f"✗ Typing Analyzer: FAILED ({e})")
        
        # Test facial analyzer
        print("\nTesting Facial Analyzer...")
        try:
            from facial_analyzer import FacialAnalyzer
            analyzer = FacialAnalyzer()
            features = analyzer.extract_features()
            test_results['facial'] = len(features) > 0
            print("✓ Facial Analyzer: PASSED")
        except Exception as e:
            test_results['facial'] = False
            print(f"✗ Facial Analyzer: FAILED ({e})")
        
        # Test wearable analyzer
        print("\nTesting Wearable Analyzer...")
        try:
            from wearable_analyzer import WearableAnalyzer
            analyzer = WearableAnalyzer()
            features = analyzer.extract_features()
            test_results['wearable'] = len(features) > 0
            print("✓ Wearable Analyzer: PASSED")
        except Exception as e:
            test_results['wearable'] = False
            print(f"✗ Wearable Analyzer: FAILED ({e})")
        
        # Test AI models
        print("\nTesting AI Models...")
        try:
            from ai_models import DataGenerator, RandomForestFatigueClassifier
            generator = DataGenerator(n_samples=100)
            X_df, y = generator.generate_training_data()
            
            model = RandomForestFatigueClassifier()
            model.train(X_df.values, y, X_df.columns.tolist())
            
            # Test prediction
            sample_features = dict(zip(X_df.columns, X_df.iloc[0]))
            prediction = model.predict(sample_features)
            
            test_results['ai_models'] = prediction.fatigue_probability is not None
            print("✓ AI Models: PASSED")
        except Exception as e:
            test_results['ai_models'] = False
            print(f"✗ AI Models: FAILED ({e})")
        
        # Test fusion system
        print("\nTesting Fusion System...")
        try:
            from fusion_system import FatigueDetectionSystem
            system = FatigueDetectionSystem()
            status = system.get_current_status()
            test_results['fusion'] = status is not None
            print("✓ Fusion System: PASSED")
        except Exception as e:
            test_results['fusion'] = False
            print(f"✗ Fusion System: FAILED ({e})")
        
        # Print summary
        print("\n" + "="*50)
        print("TEST SUMMARY")
        print("="*50)
        
        passed = sum(test_results.values())
        total = len(test_results)
        
        print(f"Tests Passed: {passed}/{total}")
        
        for test_name, result in test_results.items():
            status = "PASSED" if result else "FAILED"
            icon = "✓" if result else "✗"
            print(f"  {icon} {test_name}: {status}")
        
        return 0 if passed == total else 1
        
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Fatigue Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py web                    # Run web interface (default)
  python main.py web --port 8080        # Run on custom port
  python main.py cli --duration 120     # Run CLI mode for 2 minutes
  python main.py train                  # Train ML models
  python main.py test                   # Run system tests
        """
    )
    
    parser.add_argument(
        'mode',
        choices=['web', 'cli', 'train', 'test'],
        default='web',
        nargs='?',
        help='Operation mode (default: web)'
    )
    
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Host address for web mode (default: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port number for web mode (default: 5000)'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=60,
        help='Duration in seconds for CLI mode (default: 60)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode for web server'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = getattr(logging, args.log_level)
    setup_logging(log_level)
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Run appropriate mode
    try:
        if args.mode == 'web':
            return run_web_mode(args.host, args.port, args.debug)
        elif args.mode == 'cli':
            return run_cli_mode(args.duration)
        elif args.mode == 'train':
            return run_train_mode()
        elif args.mode == 'test':
            return run_test_mode()
        else:
            parser.print_help()
            return 1
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 0
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())