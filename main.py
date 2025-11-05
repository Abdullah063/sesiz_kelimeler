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

# === CUSTOM LAYERS - Capsule Network için ===
class Length(layers.Layer):
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1) + K.epsilon())

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = super(Length, self).get_config()
        return config


class PrimaryCapsule(layers.Layer):
    def __init__(self, num_capsule, dim_capsule, kernel_size, strides, padding, **kwargs):
        super(PrimaryCapsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

    def build(self, input_shape):
        self.conv = layers.Conv2D(
            filters=self.num_capsule * self.dim_capsule,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding
        )
        super(PrimaryCapsule, self).build(input_shape)

    def call(self, inputs):
        outputs = self.conv(inputs)
        outputs = layers.Reshape(target_shape=[-1, self.dim_capsule])(outputs)
        return self.squash(outputs)

    @staticmethod
    def squash(vectors):
        s_squared_norm = K.sum(K.square(vectors), -1, keepdims=True)
        scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
        return scale * vectors

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

    def get_config(self):
        config = {
            'num_capsule': self.num_capsule,
            'dim_capsule': self.dim_capsule,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
        }
        base_config = super(PrimaryCapsule, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings

    def build(self, input_shape):
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]
        self.W = self.add_weight(
            shape=[self.num_capsule, self.input_num_capsule,
                   self.dim_capsule, self.input_dim_capsule],
            initializer='glorot_uniform',
            name='W'
        )
        super(CapsuleLayer, self).build(input_shape)

    def call(self, inputs):
        inputs_expand = K.expand_dims(inputs, 1)
        inputs_tiled = K.tile(inputs_expand, [1, self.num_capsule, 1, 1])
        inputs_hat = K.map_fn(lambda x: K.batch_dot(x, self.W, [2, 3]), elems=inputs_tiled)

        b = tf.zeros(shape=[K.shape(inputs_hat)[0], self.num_capsule, self.input_num_capsule])

        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=1)
            outputs = self.squash(K.batch_dot(c, inputs_hat, [2, 2]))
            if i < self.routings - 1:
                b += K.batch_dot(outputs, inputs_hat, [2, 3])

        return outputs

    @staticmethod
    def squash(vectors):
        s_squared_norm = K.sum(K.square(vectors), -1, keepdims=True)
        scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
        return scale * vectors

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

    def get_config(self):
        config = {
            'num_capsule': self.num_capsule,
            'dim_capsule': self.dim_capsule,
            'routings': self.routings,
        }
        base_config = super(CapsuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# Eğer model yoksa indir
if not os.path.exists(MODEL_PATH):
    print("🔽 Model indiriliyor...")
    try:
        gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)
        print("✅ Model indirildi!")
    except Exception as e:
        print(f"❌ Model indirilemedi: {e}")
        raise

# Modeli custom objects ile yükle
print("📦 Model yükleniyor...")
custom_objects = {
    'PrimaryCapsule': PrimaryCapsule,
    'CapsuleLayer': CapsuleLayer,
    'Length': Length
}
model = load_model(MODEL_PATH, custom_objects=custom_objects)
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