from flask import Flask, request, render_template
from ultralytics import YOLO
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)

# Model yükleniyor
model = YOLO("best.pt")  # Eğittiğin YOLO model dosyası

# Klasör ayarları
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
            # Fotoğraf işle
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
                result_image=f"result/{filename}",
                baretli=baretli,
                baretsiz=baretsiz,
                kafali=kafali
            )

        elif ext in video_exts:
            # Video işle
            result_video_filename = f"result_{filename}"
            result_video_path = os.path.join(app.config['RESULT_FOLDER'], result_video_filename)

            cap = cv2.VideoCapture(file_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(result_video_path, fourcc, fps, (width, height))

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame)[0]

                baretliler = 0
                kafalilar = 0

                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id].lower()
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = (0, 255, 0) if 'helmet' in label else (0, 0, 255)

                    if 'helmet' in label:
                        baretliler += 1
                    elif 'head' in label:
                        kafalilar += 1

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                baretsiz = max(kafalilar - baretliler, 0)
                text = f"Baretli: {baretliler}  Baretsiz: {baretsiz}"

                font_scale = 2.0
                thickness = 4
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                cv2.rectangle(frame, (10, 10), (10 + text_width, 10 + text_height + baseline), (255, 255, 255), -1)
                cv2.putText(frame, text, (10, 10 + text_height), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)

                out.write(frame)

            cap.release()
            out.release()

            return render_template(
                'index.html',
                result_video=result_video_filename,
                now=datetime.now().timestamp()
            )

        else:
            return render_template('index.html', error="Desteklenmeyen dosya türü.")

    return render_template('index.html')
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, debug=True)
