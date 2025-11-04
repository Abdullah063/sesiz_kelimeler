from fastapi import FastAPI, File, UploadFile
import numpy as np
import tensorflow as tf
from PIL import Image
import io

# Uygulama başlat
app = FastAPI()

# Modeli yükle
MODEL_PATH = "models/model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Sınıf isimleri
CLASS_NAMES = [
    'N', 'R', 'space', 'B', 'I', 'del', 'F', 'H', 'E', 'U', 'M', 'K', 'Y', 'S',
    'G', 'A', 'O', 'T', 'V', 'Z', 'C', 'P', 'L', 'D', 'nothing', 'J'
]

@app.get("/")
def home():
    return {"message": "Türk İşaret Dili Model API çalışıyor 🚀"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Dosyayı oku
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((224, 224))

        # Görseli modele uygun hale getir
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Tahmin
        predictions = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        return {"predicted_class": predicted_class, "confidence": confidence}

    except Exception as e:
        return {"error": str(e)}