import cv2
import numpy as np

import torch, sys, os
sys.path.insert(0, r"C:\Users\spart\OneDrive\Documentos\Ingenieria en Desarrollo de Software\6to Cuatrimestre\Arquitectura de Computadoras\ObjectDetection\yolov7")  # ajusta la ruta
model = torch.hub.load(
    repo_or_dir=r"C:\Users\spart\OneDrive\Documentos\Ingenieria en Desarrollo de Software\6to Cuatrimestre\Arquitectura de Computadoras\ObjectDetection\yolov7", 
    model="custom", 
    source="local",
    path="models/bestYOLOv7.pt",
    trust_repo=True
)
model.eval()                       # modo inferencia
model.conf = 0.25                  # umbral de confianza
model.iou  = 0.45                  # umbral de NMS

# === 2. INICIALIZAR CÁMARA ===
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# === 3. Lógica de skip frames y dibujo ===
frame_id = 0
skip     = 2
annotated = None

while cam.isOpened():
    ret, frame = cam.read()
    if not ret:
        break

    # Sólo inferimos 1 de cada 'skip' frames
    if frame_id % skip == 0:
        # YOLOv7 espera imágenes RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Inferencia
        with torch.no_grad():
            results = model(img_rgb)

        # results.xyxy[0] es un tensor [N,6]: [x1,y1,x2,y2,conf,cls]
        detections = results.xyxy[0].cpu().numpy()

        # Copia para dibujar
        annotated = frame.copy()

        # Dibujar cada detección
        for *box, conf, cls in detections:
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(annotated, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    frame_id += 1

    # Mostrar el último frame anotado
    if annotated is not None:
        cv2.imshow("Fast YOLOv7", annotated)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
        break

cam.release()
cv2.destroyAllWindows()
