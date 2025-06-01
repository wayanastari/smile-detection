import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow import keras # Import keras dari tensorflow

# Inisialisasi detektor wajah MTCNN
detector = MTCNN()


def detect_smiles(frame, model):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(rgb_frame)

    for result in results:
        x, y, w, h = result['box']
        x, y = max(0, x), max(0, y)
        face_img = frame[y:y+h, x:x+w]

        try:
            face_resized = cv2.resize(face_img, (64, 64))

            # 1. Konversi ke grayscale
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            
            # 2. Tambahan dimensi channel. Grayscale hanya punya 1 channel.
            # Shape akan menjadi (64, 64, 1)
            face_input = np.expand_dims(face_gray, axis=-1)
            
            # 3. Tambahan dimensi batch. CNN mengharapkan input batch (batch_size, height, width, channels)
            # Shape akan menjadi (1, 64, 64, 1)
            face_input = np.expand_dims(face_input, axis=0) # atau face_input[np.newaxis, ..., np.newaxis]
            
            # 4. Normalisasi (penting jika model dilatih dengan data yang dinormalisasi)
            face_input = face_input.astype('float32') / 255.0 
    

            prediction = model.predict(face_input, verbose=0)
            confidence = float(prediction[0][0])

            label = f"Smile ({confidence:.2f})" if confidence > 0.5 else f"No Smile ({1 - confidence:.2f})"
            color = (0, 255, 0) if confidence > 0.5 else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        except Exception as e:
            print("Error during detection:", e)

    return frame