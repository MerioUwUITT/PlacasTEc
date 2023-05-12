import cv2
import numpy as np
import pytesseract   # Importa la librería pytesseract
from PIL import Image
import os
#tesseract path
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
frameWidth = 640    #Frame Width
franeHeight = 480   # Frame Height

plateCascade = cv2.CascadeClassifier("C:/Users/Mario/Documents/PlacastEk/equisemeele.xml")
minArea = 500

cap =cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

cap.set(3,frameWidth)
cap.set(4,franeHeight)
cap.set(10,150)
count = 0
while True:
    success , img  = cap.read()

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    numberPlates = plateCascade .detectMultiScale(imgGray, 1.1, 4)

    for (x, y, w, h) in numberPlates:
        area = w*h
        if area > minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img,"NumberPlate",(x,y-5),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            imgRoi = img[y:y+h,x:x+w]
            
            # Agrega el OCR de pytesseract
            img_rgb = cv2.cvtColor(imgRoi, cv2.COLOR_BGR2RGB)
            text = pytesseract.image_to_string(Image.fromarray(img_rgb), lang='eng')
            print(text)
            cv2.putText(img, text, (x,y-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.setWindowTitle("Result", text)

            # Guarda la imagen con nombre del OCR detectado
            filename = os.path.join("C:/Users/Mario/Documents/PlacastEk/Takes/", text + ".jpg")
            cv2.imwrite(filename, imgRoi)
            
            cv2.imshow("ROI",imgRoi)
    cv2.imshow("Result",img)
    if cv2.waitKey(1) & 0xFF ==ord('s'):
        
        # Quita espacios y caracteres inválidos del nombre del archivo
        text = text.strip()
        text = ''.join(e for e in text if e.isalnum())
        
        # Guarda la imagen con el nombre del OCR detectado
        filename = os.path.join("C:/Users/Mario/Documents/PlacastEk/Takes/", text + ".jpg")
        cv2.imwrite(filename, imgRoi)
        
        cv2.rectangle(img,(0,200),(640,300),(0,255,0),cv2.FILLED)
        cv2.putText(img,"Scan Saved",(15,265),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),2)
        cv2.imshow("Result",img)
        cv2.waitKey(500)
        count+=1
        
