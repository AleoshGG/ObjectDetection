import sys
import cv2
import torch
import numpy as np

# 1. Registrar safe global para numpy arrays (por si acaso)
import torch.serialization
import numpy.core.multiarray
torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct, np.ndarray])

# 2. Añadir ruta de yolov7
YV7_PATH = "/home/pybot/Documentos/ObjectDetection/yolov7"
sys.path.insert(0, YV7_PATH)

# 3. Importar la clase Model y la NMS helper
from models.yolo import Model
from utils.general import non_max_suppression

# 4. Configuración
CFG_FILE   = "cfg/training/yolov7-tiny.yaml"  # o tu cfg específica
WEIGHTS    = "models/bestYOLOv7.pt"
IMG_SIZE   = 320  # salida de cámara
CONF_THRESH = 0.25
IOU_THRESH  = 0.45
SKIP        = 2

# 5. Instanciar modelo
#    ch=3 canales, nc=len(model.names) pero lo definimos manualmente
num_classes = 2  # PET, cans-PET
model = Model(cfg=CFG_FILE, ch=3, nc=num_classes)
model.load_state_dict(
    torch.load(WEIGHTS, map_location="cpu")["model"],
    strict=False
)
model.eval()

# 6. Cámara
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH,  IMG_SIZE)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_SIZE)

frame_id = 0
annotated = None

while True:
    ret, frame = cam.read()
    if not ret:
        break

    if frame_id % SKIP == 0:
        # Preprocesar: BGR→RGB, HWC→CHW, float, normalizar, añadir batch
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = torch.from_numpy(img).permute(2,0,1).float() / 255.0
        img = img.unsqueeze(0)

        # Inferencia
        with torch.no_grad():
            pred = model(img)[0]  # [N,6] (x1,y1,x2,y2,conf,cls)

        # NMS
        pred = non_max_suppression(
            pred.unsqueeze(0), 
            CONF_THRESH, 
            IOU_THRESH
        )[0]  # retorna lista de detecciones

        annotated = frame.copy()
        if pred is not None and len(pred):
            # Scale coords de vuelta a resolución original del frame
            h0, w0 = frame.shape[:2]
            gain = min(IMG_SIZE / w0, IMG_SIZE / h0)
            pad = (0, 0)
            for *box, conf, cls in pred.cpu().numpy():
                # Deshacer scale
                x1, y1, x2, y2 = box
                x1 = int(x1 / IMG_SIZE * w0)
                x2 = int(x2 / IMG_SIZE * w0)
                y1 = int(y1 / IMG_SIZE * h0)
                y2 = int(y2 / IMG_SIZE * h0)

                label = f"{model.names[int(cls)]} {conf:.2f}"
                cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(annotated, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    frame_id += 1

    if annotated is not None:
        cv2.imshow("YOLOv7 Manual Load", annotated)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()
