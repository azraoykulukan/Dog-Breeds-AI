import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
import os

MODEL_PATH = r"D:\Azra Oyku ULUKAN\uni\4\dog_breeds\dog_breeds_cnn.h5"
TRAIN_DIR = r"D:\Azra Oyku ULUKAN\uni\4\DerinOgrenme\proje\dog-breeds"

IMG_SIZE = (128, 128)
NUM_CHANNELS = 3


CLASS_NAMES = sorted([
    d for d in os.listdir(TRAIN_DIR)
    if os.path.isdir(os.path.join(TRAIN_DIR, d))
])

print("Yüklenen sınıflar:", CLASS_NAMES)
print("Toplam sınıf:", len(CLASS_NAMES))

model = tf.keras.models.load_model(MODEL_PATH)


def predict_dog_breed(image):
    """Köpek ırkını tahmin eden fonksiyon"""
    if image is None:
        return "Lütfen bir resim yükleyin!"
    

    img = Image.fromarray(image).convert("RGB")
    img = img.resize(IMG_SIZE)
    

    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, IMG_SIZE[0], IMG_SIZE[1], NUM_CHANNELS)
    

    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    
    breed = CLASS_NAMES[class_index]
    

    top_5_idx = np.argsort(predictions[0])[-5:][::-1]
    results = {}
    for idx in top_5_idx:
        results[CLASS_NAMES[idx]] = float(predictions[0][idx])
    
    return results


# Gradio arayüzü oluştur
with gr.Blocks(title="Dog Breeds Sınıflandırma Sistemi") as demo:
    gr.Markdown("# 🐕 Dog Breeds Tanıma Sistemi")
    gr.Markdown("Bir köpek resmi yükleyin ve yapay zeka ırkını tahmin etsin!")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Köpek Resmi Yükleyin")
            predict_btn = gr.Button("Tahmin Et", variant="primary")
        
        with gr.Column():
            output_label = gr.Label(label="Tahmin Sonuçları", num_top_classes=5)
    
    predict_btn.click(
        fn=predict_dog_breed,
        inputs=image_input,
        outputs=output_label
    )
    
    gr.Markdown("### Nasıl Kullanılır?")
    gr.Markdown("1. Sol taraftan bir köpek resmi yükleyin")
    gr.Markdown("2. 'Tahmin Et' butonuna tıklayın")
    gr.Markdown("3. Sağ tarafta tahmin sonuçlarını göreceksiniz")


if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
