import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

class SensorDataExplorer:
    def __init__(self, data_folder="sensor_data"):
        """
        Initialize data explorer
        data_folder: folder containing all your CSV files
        """
        self.data_folder = data_folder
        self.activities = {
            'walking': [],
            'running': [],
            'standing': [],
            'pocket': [],
            'falls': []
        }
        
    def load_all_data(self):
        """Load all CSV files and organize by activity"""
        print("=" * 60)
        print("LOADING SENSOR DATA")
        print("=" * 60)
        
        if not os.path.exists(self.data_folder):
            print(f" Folder '{self.data_folder}' not found!")
            print(f"Please create it and put your CSV files there.")
            return False
        
        files = list(Path(self.data_folder).glob("*.csv"))
        
        if len(files) == 0:
            print(f" No CSV files found in '{self.data_folder}'")
            return False
        
        print(f"\nFound {len(files)} CSV files\n")
        
        for file in files:
            filename = file.name.lower()
            
            # Detect activity from filename
            activity = None
            if 'walk' in filename:
                activity = 'walking'
            elif 'run' in filename:
                activity = 'running'
            elif 'stand' in filename:
                activity = 'standing'
            elif 'pocket' in filename:
                activity = 'pocket'
            elif 'fall' in filename:
                activity = 'falls'  
            
            if activity:
                try:
                    df = pd.read_csv(file)
                    self.activities[activity].append({
                        'filename': file.name,
                        'data': df
                    })
                    print(f"{file.name:30s} → {activity:10s} ({len(df):6d} samples)")
                except Exception as e:
                    print(f" Error loading {file.name}: {e}")
            else:
                print(f"{file.name:30s} → Unknown activity (skip)")
        
        print("\n" + "=" * 60)
        return True
    
    def print_summary(self):
        """Print summary statistics"""
        print("\nDATA SUMMARY")
        print("=" * 60)
        
        total_samples = 0
        
        for activity, files in self.activities.items():
            if len(files) > 0:
                samples = sum([len(f['data']) for f in files])
                duration = sum([f['data']['Time (s)'].max() for f in files])
                total_samples += samples
                
                print(f"\n{activity.upper()}:")
                print(f"  Files: {len(files)}")
                print(f"  Total samples: {samples:,}")
                print(f"  Total duration: {duration:.1f} seconds")
                
                # Show sample rate
                if len(files) > 0:
                    first_df = files[0]['data']
                    if len(first_df) > 1:
                        time_diff = first_df['Time (s)'].iloc[1] - first_df['Time (s)'].iloc[0]
                        sample_rate = 1.0 / time_diff if time_diff > 0 else 0
                        print(f"  Sample rate: ~{sample_rate:.1f} Hz")
        
        print(f"\n{'TOTAL SAMPLES:'} {total_samples:,}")
        print("=" * 60)
    
    def visualize_samples(self):
        """Visualize sample data from each activity"""
        print("\nGenerating visualizations...")
        
        fig, axes = plt.subplots(5, 1, figsize=(12, 10))
        fig.suptitle('Sensor Data Comparison by Activity', fontsize=16, fontweight='bold')
        
        activity_names = ['walking', 'running', 'standing', 'pocket', 'falls']
        colors = ['blue', 'green', 'orange', 'purple', 'red']
        
        for idx, (activity, color) in enumerate(zip(activity_names, colors)):
            ax = axes[idx]
            
            if len(self.activities[activity]) > 0:
                # Take first file of this activity
                df = self.activities[activity][0]['data']
                
                # Plot only first 500 samples for clarity
                plot_data = df.head(500)
                
                # Calculate magnitude
                magnitude = np.sqrt(
                    plot_data['Acceleration x (m/s^2)']**2 + 
                    plot_data['Acceleration y (m/s^2)']**2 + 
                    plot_data['Acceleration z (m/s^2)']**2
                )
                
                ax.plot(plot_data['Time (s)'], magnitude, color=color, linewidth=1.5)
                ax.set_ylabel('Magnitude\n(m/s²)', fontsize=10)
                ax.set_title(f'{activity.upper()}', fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # Show statistics
                stats_text = f'Mean: {magnitude.mean():.2f} | Std: {magnitude.std():.2f} | Max: {magnitude.max():.2f}'
                ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
                       fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=14)
                ax.set_ylabel('Magnitude\n(m/s²)', fontsize=10)
                ax.set_title(f'{activity.upper()} - NO DATA', fontsize=11)
        
        axes[-1].set_xlabel('Time (s)', fontsize=12)
        plt.tight_layout()
        plt.savefig('sensor_data_visualization.png', dpi=150, bbox_inches='tight')
        print("Saved: sensor_data_visualization.png")
        plt.show()
    
    def check_data_quality(self):
        """Check for common data issues"""
        print("\nDATA QUALITY CHECK")
        print("=" * 60)
        
        issues_found = False
        
        for activity, files in self.activities.items():
            for file_info in files:
                df = file_info['data']
                filename = file_info['filename']
                
                # Check for missing values
                missing = df.isnull().sum().sum()
                if missing > 0:
                    print(f"{filename}: {missing} missing values")
                    issues_found = True
                
                # Check for constant values (sensor might be stuck)
                for col in ['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)']:
                    if col in df.columns:
                        if df[col].std() < 0.01:
                            print(f"{filename}: {col} appears constant (std={df[col].std():.4f})")
                            issues_found = True
                
                # Check time continuity
                if 'Time (s)' in df.columns and len(df) > 1:
                    time_diffs = df['Time (s)'].diff().dropna()
                    if time_diffs.std() > time_diffs.mean() * 0.5:
                        print(f"{filename}: Irregular sampling detected")
                        issues_found = True
        
        if not issues_found:
            print("All data looks good!")
        
        print("=" * 60)

def main():
    print("\n")
    print("SENSOR DATA EXPLORATION")
    print("\n")
    
    # Initialize explorer
    explorer = SensorDataExplorer(data_folder="sensor_data")
    
    # Load all data
    if not explorer.load_all_data():
        print("\n Failed to load data. Please check:")
        print("   1. Create a folder named 'sensor_data'")
        print("   2. Put all your CSV files in that folder")
        print("   3. Name files like: walking_001.csv, falls_001.csv, etc.")
        return
    
    # Print summary
    explorer.print_summary()
    
    # Quality check
    explorer.check_data_quality()
    
    # Visualize
    explorer.visualize_samples()
    
    print("\nExploration complete!")
    print("\nNext steps:")
    print("  1. Check 'sensor_data_visualization.png'")
    print("  2. Make sure all activities have data")
    print("  3. Ready for preprocessing!")

if __name__ == "__main__":
    main()