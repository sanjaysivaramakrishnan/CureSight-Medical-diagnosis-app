import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image


# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

MALARIA_MODEL_PATH = os.path.join(MODELS_DIR, 'malaria_cnn_model.h5')
PNEUMONIA_MODEL_PATH = os.path.join(MODELS_DIR, 'Pneumonia_model.h5')

def verify_malaria():
    print(f"Loading Malaria Model from {MALARIA_MODEL_PATH}...")
    try:
        model = load_model(MALARIA_MODEL_PATH)
        print("Malaria model loaded successfully.")
    except Exception as e:
        print(f"Failed to load Malaria model: {e}")
        return

    # Find a sample image
    # data/cell_images/Parasitized/some_image.png
    parasitized_dir = os.path.join(DATA_DIR, 'cell_images', 'Parasitized')
    if os.path.exists(parasitized_dir):
        files = os.listdir(parasitized_dir)
        if files:
            img_path = os.path.join(parasitized_dir, files[0])
            print(f"Testing on image: {img_path}")
            
            # Preprocessing needs to match training. Usually 64x64 or 128x128 or 224x224.
            # I'll check input shape from model.
            input_shape = model.input_shape[1:3] # (height, width)
            print(f"Model expects input shape: {input_shape}")
            
            img = image.load_img(img_path, target_size=input_shape)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0 # Rescale 1./255 is standard
            
            prediction = model.predict(img_array)
            print(f"Prediction: {prediction}")
            # Assuming output is probability of Uninfected or Parasitized.
            # Need to know class indices. usually 0 and 1.
    else:
        print("One of the directories not found for Malaria images.")

def verify_pneumonia():
    print(f"\nLoading Pneumonia Model from {PNEUMONIA_MODEL_PATH}...")
    try:
        model = load_model(PNEUMONIA_MODEL_PATH)
        print("Pneumonia model loaded successfully.")
    except Exception as e:
        print(f"Failed to load Pneumonia model: {e}")
        return

    # Find a sample image
    # data/chest_xray/test/PNEUMONIA/some_image.jpeg
    pneumonia_dir = os.path.join(DATA_DIR, 'chest_xray', 'test', 'PNEUMONIA')
    # If not in test, try train
    if not os.path.exists(pneumonia_dir):
         pneumonia_dir = os.path.join(DATA_DIR, 'chest_xray', 'train', 'PNEUMONIA')

    if os.path.exists(pneumonia_dir):
        files = os.listdir(pneumonia_dir)
        if files:
            img_path = os.path.join(pneumonia_dir, files[0])
            print(f"Testing on image: {img_path}")
            
            input_shape = model.input_shape[1:3]
            print(f"Model expects input shape: {input_shape}")
            
            img = image.load_img(img_path, target_size=input_shape)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0
            
            prediction = model.predict(img_array)
            print(f"Prediction: {prediction}")
    else:
        print("Directories not found for Pneumonia images.")

if __name__ == "__main__":
    verify_malaria()
    verify_pneumonia()
