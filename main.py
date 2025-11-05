from fastapi import FastAPI, File, UploadFile
import numpy as np
from PIL import Image
import io, os
import gdown
from tensorflow.keras.models import load_model

app = FastAPI()

# Model dosyası
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.h5")

# Google Drive indirme linki
FILE_ID = "1Yo-g9zbQ3YdCVSgvr_HNkaPGtVN-Iejw"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}"

# Model klasörünü oluştur
os.makedirs(MODEL_DIR, exist_ok=True)

# Eğer model yoksa indir
if not os.path.exists(MODEL_PATH):
    print("🔽 Model indiriliyor...")
    try:
        gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)
        print("✅ Model indirildi!")
    except Exception as e:
        print(f"❌ Model indirilemedi: {e}")
        raise

# Modeli yükle
print("📦 Model yükleniyor...")
model = load_model(MODEL_PATH)
print("✅ Model yüklendi!")

CLASS_NAMES = [
    'N', 'R', 'space', 'B', 'I', 'del', 'F', 'H', 'E', 'U', 'M', 'K', 'Y', 'S',
    'G', 'A', 'O', 'T', 'V', 'Z', 'C', 'P', 'L', 'D', 'nothing', 'J'
]


@app.get("/")
def home():
    return {"message": "Türk İşaret Dili Model API çalışıyor! ✨"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((224, 224))

        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        return {
            "success": True,
            "predicted_class": predicted_class,
            "confidence": confidence
        }
    except Exception as e:
        return {"success": False, "error": str(e)}