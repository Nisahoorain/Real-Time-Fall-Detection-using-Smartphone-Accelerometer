Real-Time Fall Detection using Smartphone Accelerometer
### (96.17% Accuracy | 100% Fall Precision | Zero False Alarms)

A complete end-to-end **Human Activity Recognition + Fall Detection** system using only a smartphone's built-in accelerometer sensor.

Live Demo: Phone se data → WiFi → Laptop → Real-time Fall Alert!

https://github.com/user/fall-detection-smartphone/assets/video-demo.mp4  
*(Add your recording later)*

## Features
- 5-class classification: Walking, Running, Standing, Pocket, **Falls**
- **100% Precision on Fall Detection** → Zero False Alarms!
- Real-time inference using **PhyPhox app + WiFi streaming**
- Sliding window with 50% overlap (same as training)
- Live console output + optional live visualization
- Fully reproducible preprocessing, training & inference pipeline

## Dataset & Data Collection
- Recorded using **PhyPhox Android App** (Remote Access enabled)
- Sampling Rate: **200 Hz**
- Sensor: Built-in 3-axis Accelerometer (X, Y, Z in m/s²)
- Activities performed:
  - Walking, Running, Standing still
  - Phone in pocket (natural movement)
  - Simulated falls (forward, backward, side)
- Total raw samples: ~400,000+ across 5 CSV files

## Preprocessing Pipeline (`preprocess_data.py`)
- Sliding Window: **400 samples = 2 seconds** @ 200Hz
- Overlap: **50%** (step size = 200)
- Total windows created: **1,829**
- Normalization: `StandardScaler` (fitted only on train data)
- Stratified Train-Test Split (80-20) → preserves class distribution
- Output: `(samples, 400, 3)` shape → perfect for CNN/LSTM

## Model Architecture (`train_model.py`)
**1D Convolutional Neural Network (CNN)**

| Layer                  | Details                            |
|-----------------------|------------------------------------|
| Input                 | (400, 3)                           |
| Conv1D Block 1        | 64 filters, kernel=5, ReLU + BN + Pooling + Dropout(0.3) |
| Conv1D Block 2        | 128 filters, kernel=5              |
| Conv1D Block 3        | 128 filters, kernel=3              |
| Flatten + Dense       | 256 → 128 units, ReLU              |
| Dropout               | 0.5 & 0.4                          |
| Output                | 5 classes, Softmax                 |

**Optimizer**: Adam (lr=0.001)  
**Loss**: Sparse Categorical Crossentropy  
**Callbacks**: EarlyStopping (patience=10), ReduceLROnPlateau

## Results (Test Set: 366 windows)

| Class       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Walking     | 96.23%    | 94.44% | 95.33%   | 108     |
| Running     | 95.24%    | 97.56% pris| 96.39%   | 41      |
| Standing    | 100%      | 100%   | 100%     | 79      |
| Pocket      | 91.75%    | 97.80% | 94.68%   | 91      |
| **Falls**   | **100%**  | **89.36%** | **94.38%** | 47  |

**Overall Accuracy: 96.17%**  
**Fall Detection Precision: 100% → 0 False Alarms**  
**Fall Detection Recall: 89.36% → Missed only 5 falls**

## Real-Time Inference (`real_time_inference.py`)
- Live data streaming from phone via **PhyPhox HTTP server**
- Sliding window buffer using `deque`
- Same scaler & model used as in training
- Predictions every ~0.5–1 second
- Console output with confidence
- **FALL DETECTED!** alert on fall
- Optional live matplotlib visualization

## Folder Structure
├── sensor_data/              # Raw CSV files from PhyPhox
├── processed_data/           # Preprocessed .npy + scaler + metadata
├── models/                   # Trained model (fall_detection_model.h5)
├── preprocess_data.py
├── train_model.py
├── real_time_inference.py
├── confusion_matrix.png
├── training_history.png
└── README.md
text## How to Run
1. Install PhyPhox app → Enable "Remote Access" → Note IP
2. Place your CSVs in `sensor_data/`
3. Run:
```bash
python preprocess_data.py
python train_model.py
python real_time_inference.py
Limitations & Future Improvements

Fall data is limited → can improve recall with more samples
Only accelerometer used → adding gyroscope can help
Tested on one phone → need multi-device validation
Future: Convert to TensorFlow Lite → Deploy in actual Android app