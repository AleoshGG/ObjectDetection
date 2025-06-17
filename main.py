from ultralytics import YOLO
import cv2

# Cargar el modelo
model = YOLO("models/best.pt")

# Capturar camara
cam = cv2.VideoCapture(0)

while cam.isOpened():
    ret, frame = cam.read()
    if not ret:
        break

    # Predicción del modelo
    results = model(frame)

    # Se anexan los resultados al frame mostrado y se visualiza
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Inferece", annotated_frame)

    # Codición de salida
    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()