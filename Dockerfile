# Utilizamos Python 3.9 
FROM python:3.9-slim

# Establecemos el directorio de trabajo
WORKDIR /app

# Copiamos el archivo de requerimientos
COPY requirements.txt .

# Instalamos las dependencias 
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el resto de la aplicación al contenedor
COPY . .

# Exponemos el puerto para la aplicación
EXPOSE 8000

# Comando para ejecutar la aplicación FastAPI con Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]