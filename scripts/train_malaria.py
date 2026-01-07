import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'cell_images')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'malaria_cnn_model_new.h5')

def train_malaria_model():
    print("Initializing Malaria Training...")
    
    if not os.path.exists(DATA_DIR):
        print(f"Data directory not found: {DATA_DIR}")
        return

    # Image parameters
    IMG_HEIGHT = 64 # Smaller size for malaria cells often sufficient
    IMG_WIDTH = 64
    BATCH_SIZE = 32
    EPOCHS = 5
    
    # Data Generators using validation split
    datagen = ImageDataGenerator(
        rescale=1./255, 
        validation_split=0.2,
        horizontal_flip=True,
        rotation_range=20
    )
    
    print("Loading Training Data...")
    train_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )
    
    print("Loading Validation Data...")
    validation_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )
    
    # Model Architecture
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train
    print("Starting Training...")
    model.fit(
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
    train_malaria_model()
