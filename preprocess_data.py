import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class SensorDataPreprocessor:
    def __init__(self, window_size=400, overlap=0.5):
        """
        window_size: number of samples per window (e.g., 400 = 2 seconds at 200Hz)
        overlap: overlap between windows (0.5 = 50% overlap)
        """
        self.window_size = window_size
        self.overlap = overlap
        self.step_size = int(window_size * (1 - overlap))
        self.scaler = StandardScaler()
        
        self.label_map = {
            'walking': 0,
            'running': 1,
            'standing': 2,
            'pocket': 3,
            'falls': 4
        }
        
        self.label_names = {v: k for k, v in self.label_map.items()}
        
    def load_data_from_folder(self, folder_path):
        """Load all CSV files from folder"""
        print("=" * 60)
        print("LOADING DATA")
        print("=" * 60)
        
        all_data = []
        folder = Path(folder_path)
        
        for csv_file in folder.glob("*.csv"):
            filename = csv_file.name.lower()
            
            # Determine activity label
            label = None
            for activity in self.label_map.keys():
                if activity in filename:
                    label = self.label_map[activity]
                    break
            
            if label is None:
                print(f"Skipping {csv_file.name} (unknown activity)")
                continue
            
            try:
                df = pd.read_csv(csv_file)
                
                # Extract acceleration columns
                acc_x = df['Acceleration x (m/s^2)'].values
                acc_y = df['Acceleration y (m/s^2)'].values
                acc_z = df['Acceleration z (m/s^2)'].values
                
                all_data.append({
                    'acc_x': acc_x,
                    'acc_y': acc_y,
                    'acc_z': acc_z,
                    'label': label,
                    'activity': self.label_names[label],
                    'filename': csv_file.name
                })
                
                print(f"{csv_file.name:30s} → {self.label_names[label]:10s} ({len(df):6d} samples)")
                
            except Exception as e:
                print(f" Error loading {csv_file.name}: {e}")
        
        print(f"\nLoaded {len(all_data)} files")
        print("=" * 60)
        return all_data
    
    def create_windows(self, data_list):
        """Convert time series into fixed-size windows"""
        print("\n" + "=" * 60)
        print("CREATING WINDOWS")
        print("=" * 60)
        print(f"Window size: {self.window_size} samples")
        print(f"Overlap: {self.overlap * 100:.0f}%")
        print(f"Step size: {self.step_size} samples")
        print()
        
        X = []  # Features
        y = []  # Labels
        
        for data_item in data_list:
            acc_x = data_item['acc_x']
            acc_y = data_item['acc_y']
            acc_z = data_item['acc_z']
            label = data_item['label']
            activity = data_item['activity']
            
            # Slide window through the data
            num_windows = 0
            for start in range(0, len(acc_x) - self.window_size + 1, self.step_size):
                end = start + self.window_size
                
                # Extract window
                window_x = acc_x[start:end]
                window_y = acc_y[start:end]
                window_z = acc_z[start:end]
                
                # Stack into shape (window_size, 3)
                window = np.column_stack([window_x, window_y, window_z])
                
                X.append(window)
                y.append(label)
                num_windows += 1
            
            print(f"  {activity:10s}: {num_windows} windows from {data_item['filename']}")
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"\nCreated {len(X):,} windows")
        print(f"   Shape: {X.shape}")
        print("=" * 60)
        
        return X, y
    
    def normalize_data(self, X_train, X_test):
        """Normalize features using StandardScaler"""
        print("\n" + "=" * 60)
        print("NORMALIZING DATA")
        print("=" * 60)
        
        # Reshape for scaler: (samples * window_size, features)
        n_samples, window_size, n_features = X_train.shape
        X_train_reshaped = X_train.reshape(-1, n_features)
        
        # Fit scaler on training data
        self.scaler.fit(X_train_reshaped)
        
        # Transform both train and test
        X_train_normalized = self.scaler.transform(X_train_reshaped)
        X_train_normalized = X_train_normalized.reshape(n_samples, window_size, n_features)
        
        n_samples_test, _, _ = X_test.shape
        X_test_reshaped = X_test.reshape(-1, n_features)
        X_test_normalized = self.scaler.transform(X_test_reshaped)
        X_test_normalized = X_test_normalized.reshape(n_samples_test, window_size, n_features)
        
        print(f"Normalized using StandardScaler")
        print(f"   Mean: {self.scaler.mean_}")
        print(f"   Std: {self.scaler.scale_}")
        print("=" * 60)
        
        return X_train_normalized, X_test_normalized
    
    def print_dataset_info(self, X_train, y_train, X_test, y_test):
        """Print dataset statistics"""
        print("\n" + "=" * 60)
        print("DATASET SPLIT")
        print("=" * 60)
        
        print(f"\nTraining set: {len(X_train):,} samples")
        for label in range(len(self.label_map)):
            count = np.sum(y_train == label)
            percentage = count / len(y_train) * 100
            print(f"  {self.label_names[label]:10s}: {count:5d} ({percentage:5.1f}%)")
        
        print(f"\nTest set: {len(X_test):,} samples")
        for label in range(len(self.label_map)):
            count = np.sum(y_test == label)
            percentage = count / len(y_test) * 100
            print(f"  {self.label_names[label]:10s}: {count:5d} ({percentage:5.1f}%)")
        
        print("\n" + "=" * 60)
    
    def process_and_save(self, data_folder, output_folder="processed_data", test_size=0.2):
        """Full preprocessing pipeline"""
        print("\n")
        print("SENSOR DATA PREPROCESSING PIPELINE")
        print("\n")
        
        # Create output folder
        Path(output_folder).mkdir(exist_ok=True)
        
        # Step 1: Load data
        data_list = self.load_data_from_folder(data_folder)
        
        if len(data_list) == 0:
            print(" No data loaded!")
            return
        
        # Step 2: Create windows
        X, y = self.create_windows(data_list)
        
        # Step 3: Train-test split
        print("\n" + "=" * 60)
        print("SPLITTING DATA")
        print("=" * 60)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        print(f"Split: {len(X_train)} train, {len(X_test)} test")
        print("=" * 60)
        
        # Step 4: Normalize
        X_train_norm, X_test_norm = self.normalize_data(X_train, X_test)
        
        # Print info
        self.print_dataset_info(X_train_norm, y_train, X_test_norm, y_test)
        
        # Step 5: Save processed data
        print("\n" + "=" * 60)
        print("SAVING PROCESSED DATA")
        print("=" * 60)
        
        np.save(f"{output_folder}/X_train.npy", X_train_norm)
        np.save(f"{output_folder}/X_test.npy", X_test_norm)
        np.save(f"{output_folder}/y_train.npy", y_train)
        np.save(f"{output_folder}/y_test.npy", y_test)
        
        # Save scaler and metadata
        with open(f"{output_folder}/scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        
        metadata = {
            'window_size': self.window_size,
            'overlap': self.overlap,
            'label_map': self.label_map,
            'label_names': self.label_names,
            'n_features': 3,  # x, y, z
            'n_classes': len(self.label_map)
        }
        
        with open(f"{output_folder}/metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"X_train.npy: {X_train_norm.shape}")
        print(f"X_test.npy: {X_test_norm.shape}")
        print(f"y_train.npy: {y_train.shape}")
        print(f"y_test.npy: {y_test.shape}")
        print(f"scaler.pkl")
        print(f"metadata.pkl")
        print("=" * 60)
        
        print("\nPREPROCESSING COMPLETE!")
        print(f"\nAll processed files saved in '{output_folder}/' folder")
        
        return X_train_norm, X_test_norm, y_train, y_test

def main():
    # Initialize preprocessor
    # window_size=400 means 2 seconds at 200Hz sampling rate
    preprocessor = SensorDataPreprocessor(
        window_size=400,  # 2 seconds at 200Hz
        overlap=0.5       # 50% overlap
    )
    
    # Process data
    X_train, X_test, y_train, y_test = preprocessor.process_and_save(
        data_folder="sensor_data",
        output_folder="processed_data",
        test_size=0.2
    )
    
    print("\n" + "=" * 60)
    print("READY FOR MODEL TRAINING!")
    print("=" * 60)
    print("\nNext step: Run the model training script")

if __name__ == "__main__":
    main()