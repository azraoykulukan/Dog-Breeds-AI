import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use('TkAgg')  # veya 'Agg'
import matplotlib.pyplot as plt

import os

# =========================
# AYARLAR
# =========================
DATASET_DIR = r"C:\Users\Eren\Desktop\dog-breeds" # dataset ana klasörü
IMG_SIZE = (128, 128)
BATCH_SIZE = 8
EPOCHS = 20

# =========================
# DATA AUGMENTATION & PREPROCESSING
# =========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # %20 validation
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

NUM_CLASSES = train_generator.num_classes
print("Sınıf sayısı:", NUM_CLASSES)

# =========================
# CNN MODEL
# =========================
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================
# EĞİTİM
# =========================
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# =========================
# GRAFİKLER
# =========================
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title("Accuracy")
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Loss")
plt.show()

# =========================
# MODEL KAYDET
# =========================
model.save("dog_breeds_cnn.h5")
print("✅ Model kaydedildi: dog_breeds_cnn.h5")
