import cv2
import numpy as np
import urllib.request
from tensorflow import keras

# Carica il modello addestrato
model = keras.models.load_model('modello.h5')

# Funzione per effettuare il riconoscimento dei caratteri su un'immagine
def riconosci_caratteri_immagine(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    label = np.argmax(prediction)

    return label

# Funzione per effettuare il riconoscimento dei caratteri su un file video
def riconosci_caratteri_video(video_path):
    video = cv2.VideoCapture(video_path)

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame = cv2.resize(frame, (64, 64))
        frame = frame / 255.0
        frame = np.expand_dims(frame, axis=0)

        prediction = model.predict(frame)
        label = np.argmax(prediction)

        # Mostra il risultato del riconoscimento sul frame
        cv2.putText(frame, str(label), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

# Funzione per effettuare il riconoscimento dei caratteri dalla webcam
def riconosci_caratteri_webcam():
    video = cv2.VideoCapture(0)

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame = cv2.resize(frame, (64, 64))
        frame = frame / 255.0
        frame = np.expand_dims(frame, axis=0)

        prediction = model.predict(frame)
        label = np.argmax(prediction)

        # Mostra il risultato del riconoscimento sul frame
        cv2.putText(frame, str(label), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

# Funzione per effettuare il riconoscimento dei caratteri da un'immagine URL
def riconosci_caratteri_da_url(image_url):
    resp = urllib.request.urlopen(image_url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (64, 64))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    label = np.argmax(prediction)

    return label

# Esempi di utilizzo delle funzioni

# Riconoscimento dei caratteri su un'immagine
image_path = 'test_image.jpg'
result = riconosci_caratteri_immagine(image_path)
print("Risultato riconoscimento immagine:", result)

# Riconoscimento dei caratteri su un file video
video_path = 'test_video.mp4'
riconosci_caratteri_video(video_path)

# Riconoscimento dei caratteri dalla webcam
riconosci_caratteri_webcam()

# Riconoscimento dei caratteri da un'immagine URL
image_url = 'https://example.com/image.jpg'
result = riconosci_caratteri_da_url(image_url)
print("Risultato riconoscimento immagine URL:", result)
