import numpy as np
import tensorflow as tf
import pickle
import requests
import time
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading

class RealTimeFallDetector:
    def __init__(self, model_path, metadata_path, scaler_path, phyphox_ip):
        """
        Initialize real-time fall detector
        
        Args:
            model_path: path to trained model
            metadata_path: path to metadata.pkl
            scaler_path: path to scaler.pkl
            phyphox_ip: 
        """
        print("=" * 60)
        print("INITIALIZING REAL-TIME FALL DETECTOR")
        print("=" * 60)
        
        # Load model
        print("Loading model...")
        self.model = tf.keras.models.load_model(model_path)
        print(" Model loaded")
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        print(f" Metadata loaded: {self.metadata['label_names']}")
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print(" Scaler loaded")
        
        self.phyphox_ip = phyphox_ip
        self.window_size = self.metadata['window_size']
        self.label_names = self.metadata['label_names']
        
        # Buffer for sensor data
        self.sensor_buffer = deque(maxlen=self.window_size)
        
        # Prediction history
        self.prediction_history = deque(maxlen=50)
        self.confidence_history = deque(maxlen=50)
        
        # Flag for continuous monitoring
        self.is_running = False
        
        print("=" * 60)
        print(f"Window size: {self.window_size} samples")
        print(f"Buffer size: {self.window_size} samples")
        print("=" * 60 + "\n ")
    
    def test_connection(self):
        """Test connection to Phyphox"""
        try:
            response = requests.get(f"{self.phyphox_ip}/get?acc_time&accX&accY&accZ", timeout=2)
            if response.status_code == 200:
                print(" Connected to Phyphox")
                return True
        except Exception as e:
            print(f"Connection failed: {e}")
            print("Make sure:")
            print("  1. Phyphox app is running")
            print("  2. 'Allow remote access' is enabled")
            print("  3. Phone and laptop on same WiFi")
            return False
    
    def read_sensor_data(self):
        """Read latest sensor data from Phyphox"""
        try:
            response = requests.get(
                f"{self.phyphox_ip}/get?accX&accY&accZ",
                timeout=1
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if 'buffer' in data:
                    buffers = data['buffer']
                    
                    # Get latest values
                    acc_x = buffers['accX']['buffer'][-1] if buffers['accX']['buffer'] else 0
                    acc_y = buffers['accY']['buffer'][-1] if buffers['accY']['buffer'] else 0
                    acc_z = buffers['accZ']['buffer'][-1] if buffers['accZ']['buffer'] else 0
                    
                    return [acc_x, acc_y, acc_z]
        except:
            pass
        
        return None
    
    def predict_from_window(self, window):
        """Make prediction from sensor window"""
        # Convert to numpy array
        window_array = np.array(window).reshape(1, self.window_size, 3)
        
        # Normalize
        window_normalized = self.scaler.transform(
            window_array.reshape(-1, 3)
        ).reshape(1, self.window_size, 3)
        
        # Predict
        prediction = self.model.predict(window_normalized, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        return predicted_class, confidence, prediction[0]
    
    def continuous_monitoring(self, duration=60, sample_rate=0.01):
        """
        Continuously monitor for falls
        
        Args:
            duration: monitoring duration in seconds
            sample_rate: how often to read sensors (seconds)
        """
        print("\n " + "=" * 60)
        print("STARTING CONTINUOUS FALL MONITORING")
        print("=" * 60)
        print(f"Duration: {duration} seconds")
        print(f"Sampling every: {sample_rate} seconds")
        print("\n Press Ctrl+C to stop early")
        print("=" * 60 + "\n ")
        
        start_time = time.time()
        samples_collected = 0
        predictions_made = 0
        
        try:
            while time.time() - start_time < duration:
                # Read sensor data
                sensor_data = self.read_sensor_data()
                
                if sensor_data:
                    self.sensor_buffer.append(sensor_data)
                    samples_collected += 1
                    
                    # Make prediction when buffer is full
                    if len(self.sensor_buffer) == self.window_size:
                        predicted_class, confidence, probs = self.predict_from_window(
                            list(self.sensor_buffer)
                        )
                        
                        activity = self.label_names[predicted_class]
                        
                        # Store prediction
                        self.prediction_history.append(predicted_class)
                        self.confidence_history.append(confidence)
                        
                        predictions_made += 1
                        
                        # Display prediction
                        elapsed = time.time() - start_time
                        print(f"[{elapsed:6.1f}s] {activity:10s} (confidence: {confidence:.2%})", end='')
                        
                        # Alert on fall
                        if activity == 'falls':
                            print(" FALL DETECTED!")
                        else:
                            print()
                        
                        # Clear some buffer for sliding window
                        for _ in range(self.window_size // 4):
                            self.sensor_buffer.popleft()
                
                time.sleep(sample_rate)
        
        except KeyboardInterrupt:
            print("\n \n   Monitoring stopped by user")
        
        print("\n " + "=" * 60)
        print("MONITORING SUMMARY")
        print("=" * 60)
        print(f"Duration: {time.time() - start_time:.1f} seconds")
        print(f"Samples collected: {samples_collected}")
        print(f"Predictions made: {predictions_made}")
        
        if predictions_made > 0:
            # Count predictions per activity
            print("\n Activity Distribution:")
            for label, name in self.label_names.items():
                count = self.prediction_history.count(label)
                percentage = count / predictions_made * 100
                print(f"  {name:10s}: {count:3d} ({percentage:5.1f}%)")
            
            # Average confidence
            avg_confidence = np.mean(list(self.confidence_history))
            print(f"\n Average confidence: {avg_confidence:.2%}")
        
        print("=" * 60)
    
    def live_visualization(self):
        """
        Create live visualization of predictions
        (Run this in a separate thread)
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        def update(frame):
            if len(self.prediction_history) > 0:
                # Plot prediction history
                ax1.clear()
                recent_predictions = list(self.prediction_history)[-30:]
                colors = ['blue', 'green', 'orange', 'purple', 'red']
                ax1.bar(range(len(recent_predictions)), recent_predictions, 
                       color=[colors[p] for p in recent_predictions])
                ax1.set_ylim(-0.5, 4.5)
                ax1.set_yticks(range(5))
                ax1.set_yticklabels(self.label_names.values())
                ax1.set_title('Recent Activity Predictions', fontsize=14, fontweight='bold')
                ax1.set_xlabel('Time')
                ax1.grid(axis='y', alpha=0.3)
                
                # Plot confidence
                ax2.clear()
                recent_confidence = list(self.confidence_history)[-30:]
                ax2.plot(recent_confidence, linewidth=2, color='green')
                ax2.fill_between(range(len(recent_confidence)), recent_confidence, alpha=0.3, color='green')
                ax2.set_ylim(0, 1)
                ax2.set_title('Prediction Confidence', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Time')
                ax2.set_ylabel('Confidence')
                ax2.axhline(y=0.8, color='r', linestyle='--', label='High confidence')
                ax2.legend()
                ax2.grid(alpha=0.3)
        
        ani = FuncAnimation(fig, update, interval=500, cache_frame_data=False)
        plt.tight_layout()
        plt.show()
    
    def single_prediction_demo(self):
        """Demonstrate a single prediction"""
        print("\n " + "=" * 60)
        print("SINGLE PREDICTION DEMO")
        print("=" * 60)
        print(f"Collecting {self.window_size} samples...")
        print("Move your phone to test different activities!")
        print("=" * 60 + "\n ")
        
        # Collect window
        while len(self.sensor_buffer) < self.window_size:
            sensor_data = self.read_sensor_data()
            if sensor_data:
                self.sensor_buffer.append(sensor_data)
                print(f"Progress: {len(self.sensor_buffer)}/{self.window_size}", end='\r')
                time.sleep(0.01)
        
        print("\n \n  Making prediction...")
        
        # Predict
        predicted_class, confidence, probs = self.predict_from_window(
            list(self.sensor_buffer)
        )
        
        activity = self.label_names[predicted_class]
        
        print("\n " + "=" * 60)
        print("PREDICTION RESULT")
        print("=" * 60)
        print(f"\n  Detected Activity: {activity.upper()}")
        print(f"   Confidence: {confidence:.2%}")
        
        if activity == 'falls':
            print("\n FALL DETECTED! ")
        
        print("\n All probabilities:")
        for label, name in self.label_names.items():
            bar = '█' * int(probs[label] * 50)
            print(f"  {name:10s}: {probs[label]:.2%} {bar}")
        
        print("=" * 60)

def main():
    print("\n " + "" * 20)
    print("REAL-TIME FALL DETECTION SYSTEM")
    print("" * 20 + "\n ")
    
    # Configuration
    PHYPHOX_IP = "http://192.168.100.7:8080"  #  CHANGE THIS TO YOUR IP!
    
    print("Configuration:")
    print(f"  Phyphox IP: {PHYPHOX_IP}")
    print()
    
    # Initialize detector
    detector = RealTimeFallDetector(
        model_path='models/fall_detection_model.h5',
        metadata_path='processed_data/metadata.pkl',
        scaler_path='processed_data/scaler.pkl',
        phyphox_ip=PHYPHOX_IP
    )
    
    # Test connection
    if not detector.test_connection():
        print("\n Cannot connect to Phyphox. Exiting.")
        return
    
    # Automatically start 2-minute monitoring
    print("\n  Starting 2-minute fall detection monitoring...")
    print("Press Ctrl+C to stop early\n ")
    
    detector.continuous_monitoring(duration=120)  # 2 minutes = 120 seconds
    
    print("\n  Done!")

if __name__ == "__main__":
    main()