# ⛑️ HelmetDetection: Nesne Algılama ile İş Güvenliği Denetimi

Bu proje, lisans eğitimim süresince derin öğrenme ve bilgisayarlı görü (computer vision) tekniklerini anlamak, bir yapay zeka modelini uçtan uca bir web uygulamasına dönüştürmek amacıyla geliştirilmiştir. 

## 🧠 Projenin Mantığı ve Amacı

Projenin temel odak noktası, şantiye veya fabrika gibi iş güvenliği açısından riskli alanlarda personelin kask kullanımını otomatik olarak denetlemektir. Sistemin çalışma prensibi üç ana aşamadan oluşur:

### 1. Veri İşleme ve Model Mantığı (Deep Learning)
* **YOLOv8 Mimarisi:** Projede, hız ve doğruluk dengesi nedeniyle YOLOv8 mimarisi tercih edilmiştir. Model, binlerce kasklı ve kasksız görsel ile eğitilerek (Training) nesne sınırlarını ve sınıflarını (Helmet / No-Helmet) tanımayı öğrenmiştir.
* **Ağırlık Dosyası (`best.pt`):** Eğitim sonucunda elde edilen en yüksek başarı oranına sahip model ağırlıklarıdır.

### 2. Uygulama Mimarisi (Backend & Frontend)
* **Flask Framework:** Modelin bir web arayüzü üzerinden erişilebilir olması için Python tabanlı Flask kullanılmıştır. Kullanıcıdan gelen görsel veri, modelden geçirilerek koordinatları belirlenmiş (bounding box) sonuçlar üretilir.
* **Stream & Output:** Algılanan kasklar gerçek zamanlı olarak işaretlenir ve sonuçlar web arayüzünde kullanıcıya sunulur.

### 3. Dağıtım ve Optimizasyon (DevOps)
* **Docker:** Sistemin Raspberry Pi 5 veya farklı donanımlarda "bağımlılık hatası" olmadan çalışabilmesi için tüm kütüphaneler Dockerize edilmiştir. 
* **RPi5 Entegrasyonu:** Proje, kısıtlı donanım kaynaklarında yüksek performanslı çıkarım (inference) yapabilme sınırlarını test etmek üzere optimize edilmiştir.

## 🛠️ Kullanılan Teknolojiler
* **Yapay Zeka:** YOLOv8 (Ultralytics), OpenCV
* **Backend:** Python, Flask
* **Altyapı:** Docker, Docker-Compose
* **Donanım Hedefi:** Raspberry Pi 5

## 📖 Öğrenim Kazanımları
Bu çalışma ile; bir AI modelinin Flask ile servis edilmesi, Docker ile paketlenmesi ve gerçek dünya problemlerine (iş güvenliği) yapay zeka tabanlı çözümler üretilmesi süreçlerinde deneyim kazanılmıştır.

---