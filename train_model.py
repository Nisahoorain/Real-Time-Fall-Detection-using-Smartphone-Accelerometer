import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import seaborn as sns

class FallDetectionModel:
    def __init__(self, input_shape, n_classes):
        """
        Initialize Fall Detection Model
        input_shape: (window_size, n_features) e.g., (400, 3)
        n_classes: number of activity classes (5)
        """
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.model = None
        self.history = None
        
    def build_1d_cnn(self):
        """
        Build 1D CNN architecture
        
        Architecture:
        - Conv1D layers to extract temporal features
        - MaxPooling to reduce dimensionality
        - Dropout for regularization
        - Dense layers for classification
        """
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv1D(filters=64, kernel_size=5, activation='relu', 
                         input_shape=self.input_shape, name='conv1'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2, name='pool1'),
            layers.Dropout(0.3),
            
            # Second Convolutional Block
            layers.Conv1D(filters=128, kernel_size=5, activation='relu', name='conv2'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2, name='pool2'),
            layers.Dropout(0.3),
            
            # Third Convolutional Block
            layers.Conv1D(filters=128, kernel_size=3, activation='relu', name='conv3'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2, name='pool3'),
            layers.Dropout(0.4),
            
            # Flatten and Dense layers
            layers.Flatten(),
            layers.Dense(256, activation='relu', name='fc1'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu', name='fc2'),
            layers.Dropout(0.4),
            
            # Output layer
            layers.Dense(self.n_classes, activation='softmax', name='output')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model"""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
    def print_model_summary(self):
        """Print model architecture"""
        print("\n" + "=" * 60)
        print("MODEL ARCHITECTURE - 1D CNN")
        print("=" * 60)
        self.model.summary()
        print("=" * 60)
        
    def train(self, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
        """Train the model"""
        print("\n" + "=" * 60)
        print("TRAINING MODEL")
        print("=" * 60)
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_test)}")
        print("=" * 60 + "\n")
        
        # Callbacks
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        # Train
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        print("\nTraining complete!")
        return self.history
    
    def evaluate(self, X_test, y_test, label_names):
        """Evaluate model performance"""
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)
        
        # Predictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Overall accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average=None, labels=range(len(label_names))
        )
        
        print("\nPer-Class Metrics:")
        print("-" * 60)
        print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 60)
        
        for i, label in enumerate(label_names.values()):
            print(f"{label:<12} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        print("\nConfusion Matrix:")
        print("-" * 60)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=label_names.values(),
                   yticklabels=label_names.values())
        plt.title('Confusion Matrix - Fall Detection Model', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
        print("Saved: confusion_matrix.png")
        
        # Classification report
        print("\n" + "=" * 60)
        print("DETAILED CLASSIFICATION REPORT")
        print("=" * 60)
        print(classification_report(y_test, y_pred, 
                                   target_names=label_names.values(),
                                   digits=4))
        
        # Fall detection specific metrics
        fall_label = 4  # falls class
        fall_mask_true = (y_test == fall_label)
        fall_mask_pred = (y_pred == fall_label)
        
        fall_tp = np.sum(fall_mask_true & fall_mask_pred)
        fall_fp = np.sum(~fall_mask_true & fall_mask_pred)
        fall_fn = np.sum(fall_mask_true & ~fall_mask_pred)
        fall_tn = np.sum(~fall_mask_true & ~fall_mask_pred)
        
        fall_precision = fall_tp / (fall_tp + fall_fp) if (fall_tp + fall_fp) > 0 else 0
        fall_recall = fall_tp / (fall_tp + fall_fn) if (fall_tp + fall_fn) > 0 else 0
        fall_f1 = 2 * (fall_precision * fall_recall) / (fall_precision + fall_recall) if (fall_precision + fall_recall) > 0 else 0
        
        print("\n" + "=" * 60)
        print("FALL DETECTION SPECIFIC METRICS")
        print("=" * 60)
        print(f"Fall Detection Precision: {fall_precision:.4f} (False alarms: {fall_fp})")
        print(f"Fall Detection Recall:    {fall_recall:.4f} (Missed falls: {fall_fn})")
        print(f"Fall Detection F1-Score:  {fall_f1:.4f}")
        print("=" * 60)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'fall_metrics': {
                'precision': fall_precision,
                'recall': fall_recall,
                'f1': fall_f1,
                'false_alarms': fall_fp,
                'missed_falls': fall_fn
            }
        }
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy plot
        ax1.plot(self.history.history['accuracy'], label='Train Accuracy', linewidth=2)
        ax1.plot(self.history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Loss plot
        ax2.plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        ax2.plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
        ax2.set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        print("\nSaved: training_history.png")
        plt.show()
    
    def save_model(self, filepath='fall_model.h5'):
        """Save trained model"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"\nModel saved: {filepath}")

def main():
    print("\n" + " " * 20)
    print("FALL DETECTION MODEL TRAINING")
    print(" " * 20 + "\n")
    
    # Load processed data
    print("=" * 60)
    print("LOADING PROCESSED DATA")
    print("=" * 60)
    
    X_train = np.load('processed_data/X_train.npy')
    X_test = np.load('processed_data/X_test.npy')
    y_train = np.load('processed_data/y_train.npy')
    y_test = np.load('processed_data/y_test.npy')
    
    with open('processed_data/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"Classes: {metadata['label_names']}")
    print("=" * 60)
    
    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])  # (400, 3)
    n_classes = metadata['n_classes']
    
    model = FallDetectionModel(input_shape=input_shape, n_classes=n_classes)
    model.build_1d_cnn()
    model.compile_model(learning_rate=0.001)
    model.print_model_summary()
    
    # Train model
    model.train(X_train, y_train, X_test, y_test, epochs=50, batch_size=32)
    
    # Plot training history
    model.plot_training_history()
    
    # Evaluate model
    results = model.evaluate(X_test, y_test, metadata['label_names'])
    
    # Save model
    model.save_model('models/fall_detection_model.h5')
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  confusion_matrix.png")
    print("  training_history.png")
    print("  models/fall_detection_model.h5")
    print("\nNext step: Test with real-time inference!")

if __name__ == "__main__":
    main()