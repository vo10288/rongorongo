import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

# Cartella contenente le immagini
images_folder = "immagini"
# Cartella contenente i file di testo con le coordinate dei caratteri
coordinates_folder = "coordinate"

# Dimensioni delle immagini dopo il preprocessing
image_height = 64
image_width = 64

# Caricamento dei dati
data = []
labels = []

for filename in os.listdir(coordinates_folder):
    # Carica le coordinate dei caratteri dal file di testo
    filepath = os.path.join(coordinates_folder, filename)
    coordinates = pd.read_csv(filepath)

    # Carica l'immagine corrispondente
    image_filename = os.path.splitext(filename)[0] + ".jpg"
    image_path = os.path.join(images_folder, image_filename)
    image = cv2.imread(image_path)

    # Ritaglio delle regioni di interesse (ROI)
    for index, row in coordinates.iterrows():
        x = int(row['x'])
        y = int(row['y'])
        width = int(row['width'])
        height = int(row['height'])

        roi = image[y:y+height, x:x+width]  # Ritaglio della ROI dall'immagine

        # Preprocessing delle ROI
        roi = cv2.resize(roi, (image_width, image_height))  # Ridimensiona la ROI alle dimensioni desiderate
        roi = roi / 255.0  # Normalizza i valori dei pixel nell'intervallo [0, 1]

        # Aggiungi i dati e le etichette
        data.append(roi)
        labels.append("rongorongo")

        # Preprocessing delle coordinate dei caratteri
        normalized_x = x / image.shape[1]  # Normalizza la coordinata x rispetto alla larghezza dell'immagine
        normalized_y = y / image.shape[0]  # Normalizza la coordinata y rispetto all'altezza dell'immagine
        normalized_width = width / image.shape[1]  # Normalizza la larghezza del carattere rispetto alla larghezza dell'immagine
        normalized_height = height / image.shape[0]  # Normalizza l'altezza del carattere rispetto all'altezza dell'immagine

        # Aggiungi le coordinate normalizzate
        normalized_coordinates = [normalized_x, normalized_y, normalized_width, normalized_height]

        # Aggiungi le coordinate normalizzate al dataset delle coordinate dei caratteri
        # ...

# Conversione in array NumPy
data = np.array(data)
labels = np.array(labels)

# Divisone dei dati in set di addestramento e di test
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# Parametri per l'input del modello
image_channels = 3  # Numero di canali dell'immagine (RGB)
num_classes = len(np.unique(labels))

# Creazione del modello di Machine Learning
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, image_channels)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compilazione del modello
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Addestramento del modello
model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_data=(test_data, test_labels))

# Valutazione del modello
loss, accuracy = model.evaluate(test_data, test_labels)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")
