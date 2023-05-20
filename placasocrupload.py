import cv2
import numpy as np
import pytesseract
import tkinter as tk
from tkinter import filedialog
import os

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
plateCascade = cv2.CascadeClassifier("C:/Users/Mario/Documents/PlacastEk/equisemeele.xml")
minArea = 500

# Abrir la ventana de selección de archivos
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()

# Cargar la imagen de la placa
image = cv2.imread(file_path)

# Preprocesar la imagen
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# Detectar la placa
numberPlates = plateCascade.detectMultiScale(gray, 1.1, 4)
for (x, y, w, h) in numberPlates:
    area = w*h
    if area > minArea:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        imgRoi = image[y:y+h,x:x+w]

        # Reconocer los caracteres alfanuméricos en el ROI
        img_rgb = cv2.cvtColor(imgRoi, cv2.COLOR_BGR2RGB)
        text = pytesseract.image_to_string(img_rgb, lang='eng', config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        text = text.replace('\n', '').replace('\r', '') # Eliminar saltos de línea y retornos de carro

        # Guardar la imagen del ROI con el nombre del texto OCR detectado
        filename = os.path.join("C:/Users/Mario/Documents/PlacastEk/Takes", text + ".jpg")
        print("Guardando imagen como:", filename) # Mensaje de depuración
        cv2.imwrite(filename, imgRoi)

        # Comprobar si el archivo existe y mostrar un mensaje de confirmación
        if os.path.exists(filename):
            print("La imagen ha sido guardada correctamente como", filename)

        # Mostrar el texto OCR detectado en la ventana de imagen
        cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Mostrar la imagen con la placa y los caracteres alfanuméricos detectados
cv2.imshow('Imagen', image)
while True:
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

# Cerrar todas las ventanas abiertas
cv2.destroyAllWindows()

# Cerrar el programa
exit()
