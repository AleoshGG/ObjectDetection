import threading, cv2, torch
from ultralytics import YOLO

model = YOLO("models/YOLOv11.pt")  # o donde tengas tu modelo entrenado
model.fuse()  # optimización opcional

frame_lock = threading.Lock()
latest_frame = None

def capture_thread():
    global latest_frame
    cap = cv2.VideoCapture(0)
    cap.set(3,320); cap.set(4,240)
    while True:
        ret, f = cap.read()
        if not ret: break
        with frame_lock:
            latest_frame = f

t = threading.Thread(target=capture_thread, daemon=True)
t.start()

while True:
    with frame_lock:
        f = latest_frame.copy() if latest_frame is not None else None
    if f is None: continue

    # reescala e infiere
    small = cv2.resize(f, (256,256))
    results = model(small)  # batch de 1
    ann = results[0].plot()
    img = cv2.resize(ann, (320,240))

    cv2.imshow("Async Fast YOLOv11", img)
    if cv2.waitKey(1)==27: break
