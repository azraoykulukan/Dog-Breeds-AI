import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import os

# ==============================
# MODEL & DATASET
# ==============================
MODEL_PATH = r"C:\Users\Eren\PycharmProjects\dog_breeds\dog_breeds_cnn.h5"
TRAIN_DIR = r"C:\Users\Eren\Desktop\dog-breeds"  # eğitim klasöründeki her sınıf alt klasörleri

IMG_SIZE = (128, 128)   # eğitimde kullandığın boyut
NUM_CHANNELS = 3        # RGB

# ==============================
# SINIFLARI OTOMATİK AL
# ==============================
CLASS_NAMES = sorted([
    d for d in os.listdir(TRAIN_DIR)
    if os.path.isdir(os.path.join(TRAIN_DIR, d))
])

print("Yüklenen sınıflar:", CLASS_NAMES)
print("Toplam sınıf:", len(CLASS_NAMES))

# ==============================
# MODEL YÜKLE
# ==============================
model = tf.keras.models.load_model(MODEL_PATH)

# ==============================
# TAHMİN FONKSİYONU
# ==============================
def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(IMG_SIZE)

    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, IMG_SIZE[0], IMG_SIZE[1], NUM_CHANNELS)

    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    return CLASS_NAMES[class_index], confidence

# ==============================
# BUTON FONKSİYONU
# ==============================
def upload_and_predict():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
    )

    if not file_path:
        return

    img = Image.open(file_path)
    img_resized = img.resize((200, 200))
    img_tk = ImageTk.PhotoImage(img_resized)

    image_label.config(image=img_tk)
    image_label.image = img_tk

    label, confidence = predict_image(file_path)

    result_label.config(
        text=f"Tahmin: {label}\nGüven: %{confidence:.2f}"
    )

# ==============================
# TKINTER ARAYÜZ
# ==============================
root = tk.Tk()
root.title("Dog Breeds Sınıflandırma Sistemi")
root.geometry("420x480")

title = tk.Label(
    root,
    text="Dog Breeds Tanıma",
    font=("Arial", 16, "bold")
)
title.pack(pady=10)

btn = tk.Button(
    root,
    text="Resim Yükle",
    command=upload_and_predict,
    font=("Arial", 12)
)
btn.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=10)

result_label = tk.Label(
    root,
    text="",
    font=("Arial", 12)
)
result_label.pack(pady=10)

root.mainloop()
