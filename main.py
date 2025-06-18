from ultralytics import YOLO
import cv2

# 1. Modelo más ligero
model = YOLO("models/best.pt")
model.fuse()  # fusiona BatchNorm+Conv

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

frame_id = 0
skip = 2

while cam.isOpened():
    ret, frame = cam.read()
    if not ret: break

    if frame_id % skip == 0:
        # inferencia asíncrona (stream=True)
        for results in model.track(frame, stream=True):
            annotated = results.plot()
            break
    frame_id += 1

    cv2.imshow("Fast YOLOv8", annotated)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()
