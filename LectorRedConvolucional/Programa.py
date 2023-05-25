import cv2
import numpy as np
from keras.models import load_model

# Cargar el modelo entrenado para la clasificación de dígitos
model = load_model('modelo1.h5')

# Cargar la imagen y convertirla a escala de grises
img = cv2.imread('placas.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Aplicar un umbral adaptativo para binarizar la imagen
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Encontrar los contornos de los objetos en la imagen
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop a través de los contornos
for cnt in contours:
    # Obtener la coordenada del rectángulo que encierra el contorno
    x, y, w, h = cv2.boundingRect(cnt)
    
    # Recortar el contorno de la imagen original y redimensionarla para que sea compatible con la entrada del modelo
    digit = gray[y:y+h, x:x+w]
    resized_digit = cv2.resize(digit, (28,28))
    normalized_digit = resized_digit / 255.0
    
    # Aplanar la imagen en un vector de una dimensión para que sea compatible con la entrada del modelo
    flattened_digit = normalized_digit.reshape(1, 28, 28, 1)
    
    # Utilizar el modelo para predecir el dígito en la imagen
    prediction = model.predict(flattened_digit)
    digit_class = np.argmax(prediction)
    
    # Dibujar un rectángulo alrededor del dígito detectado y mostrar la predicción en la imagen
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    cv2.putText(img, str(digit_class), (x+int(w/2),y-int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    
# Mostrar la imagen con los dígitos detectados y sus predicciones
cv2.imshow('Resultado', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
