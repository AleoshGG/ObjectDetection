import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

# === CONFIGURACIÓN ===
MODEL_PATH = "models/best_int8.tflite"
CLASSES = ["PET", "cans-PET"]
INPUT_SIZE = 640  # usa 640 si tu Pi aguanta sin lag

# === INICIAR MODELO ===
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_index = input_details[0]['index']
output_index = output_details[0]['index']

# === CAPTURA DE CÁMARA ===
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, INPUT_SIZE)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, INPUT_SIZE)

# CREA UNA VENTANA FIJA (una sola vez)
window_name = "Detección PET y Aluminio"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)


while True:
    ret, frame = cam.read()
    if not ret:
        break

    # PREPROCESAR IMAGEN
    image = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    input_data = np.expand_dims(image, axis=0).astype(np.float32)
    input_data = input_data / 255.0  # Normaliza a rango [0, 1]


    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    # OBTENER RESULTADOS
    output_data = interpreter.get_tensor(output_index)[0]

    # === INTERPRETAR DETECCIONES ===
    for det in output_data:
        if det[4] < 0.5:
            continue  # umbral de confianza

        x1, y1, x2, y2 = (det[0:4] * INPUT_SIZE).astype(int)
        class_id = int(det[5])
        label = CLASSES[class_id] if class_id < len(CLASSES) else "???"

        # DIBUJAR CAJA
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # MOSTRAR RESULTADO
    cv2.imshow(window_name, image)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()
