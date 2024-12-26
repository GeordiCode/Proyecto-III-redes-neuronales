from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
from io import BytesIO
from fastapi import Request
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI()

# Configura la ruta estática para acceder a archivos como CSS, JS, etc.
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# Cargar el modelo previamente entrenado
model = load_model("modelo_denso_weather_data1.h5")

# Diccionario de categorías
categories = ["dew", "fogsmog", "frost", "glaze", "hail", "lightning", "rain", "rainbow", "rime", "sandstorm", "snow"]

# Configurar templates
templates = Jinja2Templates(directory="templates")

# Ruta principal con formulario
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Ruta para hacer predicción
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Leer la imagen
        image = Image.open(file.file).convert("RGB")
        image = image.resize((64, 64))  # Tamaño esperado por el modelo
        image = img_to_array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        # Hacer predicción
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction[0])

        # Obtener el nombre de la categoría
        predicted_category = categories[predicted_class]

        return {"category": predicted_category, "confidence": float(prediction[0][predicted_class])}
    except Exception as e:
        return {"error": str(e)}
