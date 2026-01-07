import os
import numpy as np
from PIL import Image
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Dropout
from keras.models import Sequential
from keras import backend as K
from sklearn.model_selection import train_test_split

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The data has nested structure: data/cell_images/cell_images/Parasitized
DATA_DIR = os.path.join(BASE_DIR, 'data', 'cell_images', 'cell_images')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'malaria_cnn_model_fast.h5')

def CNNbuild(height, width, classes, channels):
    """Build CNN model matching the original Kaggle notebook architecture"""
    model = Sequential()
    
    inputShape = (height, width, channels)
    chanDim = -1
    
    if K.image_data_format() == 'channels_first':
        inputShape = (channels, height, width)
    
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=inputShape))
    model.add(MaxPooling2D(2,2))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Dropout(0.2))

    model.add(Flatten())
    
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax'))
    
    return model

def train_malaria():
    print("Initializing Malaria Model Training...")
    print(f"Data directory: {DATA_DIR}")
    
    # Check if data exists
    parasitized_path = os.path.join(DATA_DIR, 'Parasitized')
    uninfected_path = os.path.join(DATA_DIR, 'Uninfected')
    
    if not os.path.exists(parasitized_path):
        print(f"ERROR: Parasitized folder not found at {parasitized_path}")
        return
    if not os.path.exists(uninfected_path):
        print(f"ERROR: Uninfected folder not found at {uninfected_path}")
        return
    
    parasitized_data = [f for f in os.listdir(parasitized_path) if f.endswith('.png')]
    uninfected_data = [f for f in os.listdir(uninfected_path) if f.endswith('.png')]
    
    print(f"Found {len(parasitized_data)} parasitized images")
    print(f"Found {len(uninfected_data)} uninfected images")
    
    # Load and preprocess data
    data = []
    labels = []
    
    print("Loading parasitized images...")
    count = 0
    for img_name in parasitized_data:
        try:
            img_path = os.path.join(parasitized_path, img_name)
            # Use PIL to load and resize
            img = Image.open(img_path).convert('RGB')
            img = img.resize((50, 50))
            img_array = img_to_array(img)
            data.append(img_array)
            labels.append(1)  # Parasitized = 1
            count += 1
            if count % 2000 == 0:
                print(f"  Loaded {count} parasitized images...")
        except Exception as e:
            pass
    
    print(f"Loaded {count} parasitized images")
    
    print("Loading uninfected images...")
    count = 0
    for img_name in uninfected_data:
        try:
            img_path = os.path.join(uninfected_path, img_name)
            # Use PIL to load and resize
            img = Image.open(img_path).convert('RGB')
            img = img.resize((50, 50))
            img_array = img_to_array(img)
            data.append(img_array)
            labels.append(0)  # Uninfected = 0
            count += 1
            if count % 2000 == 0:
                print(f"  Loaded {count} uninfected images...")
        except Exception as e:
            pass
    
    print(f"Loaded {count} uninfected images")
    
    # Convert to numpy arrays
    image_data = np.array(data)
    labels = np.array(labels)
    
    print(f"Total images: {len(image_data)}")
    print(f"Image shape: {image_data[0].shape if len(image_data) > 0 else 'N/A'}")
    
    # Shuffle data
    idx = np.arange(image_data.shape[0])
    np.random.shuffle(idx)
    image_data = image_data[idx]
    labels = labels[idx]
    
    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        image_data, labels, test_size=0.2, random_state=101
    )
    
    # Convert labels to categorical (one-hot encoding)
    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)
    
    # Normalize pixel values
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    print(f'Training data shape: {x_train.shape}')
    print(f'Testing data shape: {x_test.shape}')
    
    # Build model
    height = 50
    width = 50
    classes = 2
    channels = 3
    
    model = CNNbuild(height=height, width=width, classes=classes, channels=channels)
    model.summary()
    
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    
    # Train model
    print("\nStarting Training...")
    h = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
    
    # Evaluate
    print("\nEvaluating model...")
    predictions = model.evaluate(x_test, y_test)
    print(f'Test Loss: {predictions[0]:.4f}')
    print(f'Test Accuracy: {predictions[1]:.4f}')
    
    # Save model
    print(f"\nSaving model to {MODEL_PATH}...")
    model.save(MODEL_PATH)
    print("Done!")

if __name__ == "__main__":
    train_malaria()
