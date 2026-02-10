# Raspberry Pi 5 (ARM64) uyumlu Dockerfile
FROM python:3.11-slim

# Sistem bağımlılıklarını yükle (OpenCV ve Ultralytics için gerekli)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Çalışma dizinini ayarla
WORKDIR /app

# Bağımlılıkları kopyala ve yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Proje dosyalarını kopyala
COPY . .

# Çıkış portunu ayarla
EXPOSE 8082

# Uygulamayı çalıştır
CMD ["python", "app.py"]
