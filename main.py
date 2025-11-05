from fastapi import FastAPI, File, UploadFile
import numpy as np
from PIL import Image
import io, os
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import tensorflow as tf

app = FastAPI()

# Model dosyası
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.h5")

# Google Drive indirme linki
FILE_ID = "1Yo-g9zbQ3YdCVSgvr_HNkaPGtVN-Iejw"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}"

# Model klasörünü oluştur
os.makedirs(MODEL_DIR, exist_ok=True)

# === CUSTOM LAYERS - Eğitim koduyla aynı ===

# Squash fonksiyonu
def squash(vectors, axis=-1):
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm)
    return scale * vectors / K.sqrt(s_squared_norm + K.epsilon())


# Primary Capsule katmanı
class PrimaryCapsule(layers.Layer):
    def __init__(self, dim_capsule, n_channels, kernel_size, strides, padding, **kwargs):
        super(PrimaryCapsule, self).__init__(**kwargs)
        self.dim_capsule = dim_capsule
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.output_num_capsule = None

    def build(self, input_shape):
        self.conv = layers.Conv2D(
            filters=self.dim_capsule * self.n_channels,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            name='primary_conv'
        )
        self.conv.build(input_shape)
        
        conv_output_shape = self.conv.compute_output_shape(input_shape)
        
        if conv_output_shape[1] is not None and conv_output_shape[2] is not None:
            self.output_num_capsule = conv_output_shape[1] * conv_output_shape[2] * self.n_channels
        
        super(PrimaryCapsule, self).build(input_shape)
    
    def compute_output_shape(self, input_shape):
        conv_output_shape = self.conv.compute_output_shape(input_shape)
        
        if conv_output_shape[1] is not None and conv_output_shape[2] is not None:
            num_capsule = conv_output_shape[1] * conv_output_shape[2] * self.n_channels
            return (input_shape[0], num_capsule, self.dim_capsule)
        else:
            return (input_shape[0], None, self.dim_capsule)

    def call(self, inputs):
        outputs = self.conv(inputs)
        batch_size = tf.shape(outputs)[0]
        outputs = K.reshape(outputs, [batch_size, -1, self.dim_capsule])
        return squash(outputs)
    
    def get_config(self):
        config = {
            'dim_capsule': self.dim_capsule,
            'n_channels': self.n_channels,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding
        }
        base_config = super(PrimaryCapsule, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# Kapsül katmanı
class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.input_num_capsule = None
        self.input_dim_capsule = None

    def build(self, input_shape):
        if input_shape[1] is not None:
            self.input_num_capsule = int(input_shape[1])
        else:
            raise ValueError(f"Input shape[1] cannot be None. Got input_shape: {input_shape}")
            
        if input_shape[2] is not None:
            self.input_dim_capsule = int(input_shape[2])
        else:
            raise ValueError(f"Input shape[2] cannot be None. Got input_shape: {input_shape}")
        
        self.W = self.add_weight(
            shape=[self.num_capsule, self.input_num_capsule, self.dim_capsule, self.input_dim_capsule],
            initializer='glorot_uniform',
            trainable=True,
            name='W'
        )
            
        super(CapsuleLayer, self).build(input_shape)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_capsule, self.dim_capsule)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        
        inputs_expand = K.expand_dims(inputs, axis=1)
        inputs_tile = K.tile(inputs_expand, [1, self.num_capsule, 1, 1])
        inputs_tile_expanded = K.expand_dims(inputs_tile, axis=3)
        
        W_tiled = K.expand_dims(self.W, axis=0)
        
        inputs_hat = tf.reduce_sum(inputs_tile_expanded * W_tiled, axis=-1)
        
        b = tf.zeros([batch_size, self.num_capsule, self.input_num_capsule])
        
        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=1)
            c_expand = K.expand_dims(c, axis=-1)
            
            s = tf.reduce_sum(c_expand * inputs_hat, axis=2)
            outputs = squash(s, axis=-1)
            
            if i < self.routings - 1:
                outputs_expand = K.expand_dims(outputs, axis=2)
                agreement = tf.reduce_sum(inputs_hat * outputs_expand, axis=-1)
                b = b + agreement
        
        return outputs
    
    def get_config(self):
        config = {
            'num_capsule': self.num_capsule,
            'dim_capsule': self.dim_capsule,
            'routings': self.routings
        }
        base_config = super(CapsuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# Length katmanı (eğitim kodunda Lambda olarak kullanılmış)
class Length(layers.Layer):
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1) + K.epsilon())

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = super(Length, self).get_config()
        return config


# Model indir
if not os.path.exists(MODEL_PATH):
    print("🔽 Model indiriliyor...")
    try:
        gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)
        print("✅ Model indirildi!")
    except Exception as e:
        print(f"❌ Model indirilemedi: {e}")
        raise

# Modeli yükle - CUSTOM OBJECTS İLE
print("📦 Model yükleniyor...")
custom_objects = {
    'PrimaryCapsule': PrimaryCapsule,
    'CapsuleLayer': CapsuleLayer,
    'Length': Length,
    'squash': squash  # ÖNEMLİ: squash fonksiyonunu da ekle
}

try:
    model = load_model(MODEL_PATH, custom_objects=custom_objects)
    print("✅ Model başarıyla yüklendi!")
except Exception as e:
    print(f"❌ Model yükleme hatası: {e}")
    raise

# Sınıf isimleri - Eğitim kodundaki sırayla
CLASS_NAMES = [
    'N', 'R', 'space', 'B', 'I', 'del', 'F', 'H', 'E', 'U', 'M', 'K', 'Y', 'S',
    'G', 'A', 'O', 'T', 'V', 'Z', 'C', 'P', 'L', 'D', 'nothing', 'J'
]


@app.get("/")
def home():
    return {
        "message": "Türk İşaret Dili Model API çalışıyor! ✨",
        "model_loaded": True,
        "classes": len(CLASS_NAMES),
        "version": "1.0"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "total_classes": len(CLASS_NAMES)
    }


@app.get("/classes")
def get_classes():
    """Tüm sınıfları listele"""
    return {
        "classes": CLASS_NAMES,
        "total": len(CLASS_NAMES)
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Resim yükleyerek tahmin yap"""
    try:
        # Resmi oku ve işle
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((224, 224))

        # Model için hazırla
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Tahmin yap
        predictions = model.predict(img_array, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(np.max(predictions[0]))

        # Top 3 tahmin
        top3_idx = np.argsort(predictions[0])[-3:][::-1]
        top3_predictions = [
            {
                "class": CLASS_NAMES[idx],
                "confidence": float(predictions[0][idx])
            }
            for idx in top3_idx
        ]

        return {
            "success": True,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "top3_predictions": top3_predictions
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """Birden fazla resim için tahmin yap"""
    results = []
    
    for file in files:
        try:
            contents = await file.read()
            img = Image.open(io.BytesIO(contents)).convert("RGB")
            img = img.resize((224, 224))

            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array, verbose=0)
            predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
            confidence = float(np.max(predictions[0]))

            results.append({
                "filename": file.filename,
                "success": True,
                "predicted_class": predicted_class,
                "confidence": confidence
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {"results": results}