from ultralytics import YOLO

# Eğittiğin modeli yükle
model = YOLO("best.pt")

# Tahmin yap (bir görsel dosyasıyla)
results = model.predict(source="2.jpg", show=True, save=True)