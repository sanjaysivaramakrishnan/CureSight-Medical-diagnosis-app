import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'chest_xray')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'pneumonia_model_new.h5')

def train_pneumonia_model():
    print("Initializing Pneumonia Training...")
    
    # Image parameters
    IMG_HEIGHT = 150
    IMG_WIDTH = 150
    BATCH_SIZE = 32
    EPOCHS = 5 # Small number for demonstration/speed
    
    train_dir = os.path.join(DATA_DIR, 'train')
    val_dir = os.path.join(DATA_DIR, 'val')
    test_dir = os.path.join(DATA_DIR, 'test')
    
    if not os.path.exists(train_dir):
        print(f"Train directory not found: {train_dir}")
        return

    # Data Generators
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    print("Loading Training Data...")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    
    print("Loading Validation Data...")
    # Using test dir as validation if val is too small (sometimes val is extremely small in this dataset)
    # Actually val is often just 16 images. Let's use test for validation or split train.
    # The dataset typically has train/test/val.
    validation_generator = val_datagen.flow_from_directory(
        test_dir, # Using test for validation to have better metrics during training
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    
    # Model Architecture
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train
    print("Starting Training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE
    )
    
    # Save
    print(f"Saving model to {MODEL_PATH}...")
    model.save(MODEL_PATH)
    print("Done.")

if __name__ == "__main__":
    train_pneumonia_model()
