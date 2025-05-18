from flask import Flask, request, render_template, Response
from ultralytics import YOLO
import os
import cv2
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)

# Model yükleniyor
model = YOLO("best.pt")  # Kendi eğittiğin YOLOv11 modelinin yolu

# Klasörler
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/result'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="Dosya seçilmedi.")

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="Dosya seçilmedi.")

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        ext = filename.split('.')[-1].lower()
        image_exts = ['jpg', 'jpeg', 'png']
        video_exts = ['mp4', 'avi', 'mov', 'mkv']

        if ext in image_exts:
            # Görsel işleme
            img = cv2.imread(file_path)
            results = model(img)[0]

            baretli = 0
            kafali = 0

            for box in results.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id].lower()
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (0, 255, 0) if 'helmet' in label else (0, 0, 255)

                if 'helmet' in label:
                    baretli += 1
                elif 'head' in label:
                    kafali += 1

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            baretsiz = max(kafali - baretli, 0)
            result_path = os.path.join(app.config['RESULT_FOLDER'], filename)
            cv2.imwrite(result_path, img)

            return render_template(
                'index.html',
                result_image=filename,  # Sadece dosya adı
                baretli=baretli,
                baretsiz=baretsiz,
)

        elif ext in video_exts:
            # Video yüklendi ve analiz edilecek (canlı gösterim için yönlendirme)
            return render_template('index.html', result_video=filename)

        else:
            return render_template('index.html', error="Desteklenmeyen dosya türü.")

    return render_template('index.html')


# Anlık video akışını sağlayan fonksiyon
def generate_frames(path):
    cap = cv2.VideoCapture(path)

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)[0]
        baretli = 0
        kafali = 0

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id].lower()
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = (0, 255, 0) if 'helmet' in label else (0, 0, 255)

            if 'helmet' in label:
                baretli += 1
            elif 'head' in label:
                kafali += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        baretsiz = max(kafali - baretli, 0)
        text = f"Baretli: {baretli}  Baretsiz: {baretsiz}"
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


# Video akışı için route
@app.route('/video_feed/<filename>')
def video_feed(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return Response(generate_frames(video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, debug=True)
