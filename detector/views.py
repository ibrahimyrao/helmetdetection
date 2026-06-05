import os
import cv2
import json
from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from ultralytics import YOLO

# Model yükleniyor (Global bir şekilde bir kez yüklenir)
MODEL_PATH = os.path.join(settings.BASE_DIR, 'best.pt')
model = YOLO(MODEL_PATH)

def index(request):
    context = {}
    if request.method == 'POST' and request.FILES.get('file'):
        myfile = request.FILES['file']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        file_path = fs.path(filename)
        
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
            
            # Sonucu kaydet
            result_filename = f"result_{filename}"
            result_path = os.path.join(settings.MEDIA_ROOT, result_filename)
            cv2.imwrite(result_path, img)

            context = {
                'result_image': f"{settings.MEDIA_URL}{result_filename}",
                'baretli': baretli,
                'baretsiz': baretsiz,
            }

        elif ext in video_exts:
            # Video stream için yönlendirme
            context = {'result_video': filename}

        else:
            context = {'error': "Desteklenmeyen dosya türü."}

    return render(request, 'index.html', context)

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

def video_feed(request, filename):
    video_path = os.path.join(settings.MEDIA_ROOT, filename)
    return StreamingHttpResponse(generate_frames(video_path),
                                 content_type='multipart/x-mixed-replace; boundary=frame')
