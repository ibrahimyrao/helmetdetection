# ⛑️ HelmetDetection: Yapay Zeka ile İş Güvenliği Denetimi

Bu proje, şantiye, fabrika ve endüstriyel sahalarda iş güvenliğinin otomatik olarak denetlenmesini sağlamak amacıyla geliştirilmiş, derin öğrenme tabanlı bir kask/baret algılama sistemidir. Görseller ve video akışları üzerinde kask kullanımını gerçek zamanlı olarak tespit eder.

---

## 🧠 Proje Mimarisi ve Çalışma Mantığı

Sistem, yapay zeka çıkarımları ile web sunucu teknolojilerini bir araya getiren üç temel katmandan oluşur:

### 1. Derin Öğrenme & Nesne Algılama (AI/Vision)
*   **YOLOv8 (Ultralytics):** Nesne tespiti görevinde hız ve doğruluk dengesi nedeniyle YOLOv8 mimarisi tercih edilmiştir. Model, iş güvenliği baretleri (Helmet) ve çıplak kafaları (No-Helmet) tanımak üzere özel olarak eğitilmiştir.
*   **Model Ağırlığı (`best.pt`):** Eğitim aşaması sonucunda elde edilen, en yüksek hassasiyet ve doğruluk oranına sahip ağırlık dosyasıdır.
*   **OpenCV (Headless):** Görüntü işleme, video akışı ayrıştırma ve tespit edilen nesneleri görselleştirmek (bounding box çizimleri) için kullanılır.

### 2. Uygulama Katmanı (Backend & Frontend)
*   **Django Web Framework:** Uygulamanın kontrol paneli, dosya yükleme süreçleri ve yönlendirmeleri Django ile yapılandırılmıştır.
*   **Statik ve Medya Yönetimi:** Yüklenen dosyalar `media/` klasöründe işlenir. Statik dosyalar production ortamında hızlı sunum için WhiteNoise ile servis edilir.
*   **Modern Arayüz:** Kullanıcı dostu, mesh gradyanlar ve dinamik aydınlatma efektlerine sahip modern, karanlık mod odaklı bir web arayüzü sunar.

### 3. Dağıtım ve Altyapı (DevOps)
*   **Docker & Docker Compose:** Projenin bağımlılık uyuşmazlığı olmadan Raspberry Pi 5 veya bulut sunucular üzerinde çalıştırılabilmesi amacıyla tamamen Dockerize edilmiştir.

---

## 🛠️ Kullanılan Teknolojiler

*   **Yapay Zeka:** YOLOv8 (Ultralytics), OpenCV
*   **Backend:** Python 3, Django
*   **Sunucu & Statik:** Gunicorn, WhiteNoise
*   **Altyapı:** Docker & Docker Compose

---

## 🚀 Hızlı Başlangıç

### 1. Yerel Kurulum (Geliştirme Ortamı)

**Gereksinimler:** Python 3.10+, pip, virtualenv

1.  **Projeyi Klonlayın:**
    ```bash
    git clone https://github.com/KULLANICI_ADINIZ/helmetdetection.git
    cd helmetdetection
    ```

2.  **Sanal Ortam Oluşturun ve Aktifleştirin:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Windows için: venv\Scripts\activate
    ```

3.  **Gerekli Paketleri Yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Ortam Değişkenlerini Oluşturun:**
    Proje ana dizininde bir `.env` dosyası oluşturun ve içerisine şunları ekleyin:
    ```env
    SECRET_KEY=kendi-guvenli-anahtariniz
    DEBUG=True
    ALLOWED_HOSTS=localhost,127.0.0.1
    CSRF_TRUSTED_ORIGINS=http://localhost:8000
    ```

5.  **Veritabanını Migre Edin:**
    ```bash
    python manage.py migrate
    ```

6.  **Uygulamayı Başlatın:**
    ```bash
    python manage.py runserver
    ```
    Uygulamaya tarayıcınızdan `http://127.0.0.1:8000` adresinden erişebilirsiniz.

---

### 🐳 2. Docker ile Çalıştırma

Docker compose kullanarak tüm bağımlılıkları içeren konteyneri tek komutla ayağa kaldırabilirsiniz:

1.  **Konteyneri Derleyin ve Başlatın:**
    ```bash
    docker compose up -d --build
    ```

2.  **Veritabanı Kurulumunu Tamamlayın:**
    ```bash
    docker compose exec helmet-tracker python manage.py migrate
    ```

Uygulama varsayılan olarak **8083** portunda çalışacaktır (`http://localhost:8083`).

---

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Daha fazla bilgi için `LICENSE` dosyasına göz atabilirsiniz.