import cv2
import pytesseract

# Capturar la imagen de la placa
image = cv2.imread('placa3.png')

# Preprocesar la imagen
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# Segmentar los dígitos y letras
cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
char_cnts = []
for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    if w >= 10 and h >= 10:
        aspect_ratio = w / float(h)
        if 0.5 <= aspect_ratio <= 1.5:
            char_cnts.append(c)
char_cnts = sorted(char_cnts, key=lambda x: cv2.boundingRect(x)[0])

# Reconocer los dígitos y letras
chars = ''
for c in char_cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    roi = gray[y:y + h, x:x + w]
    config = '-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8'
    text = pytesseract.image_to_string(roi, config=config)
    chars += text.strip()
    
cv2.namedWindow("Contornos")
cv2.drawContours(image, char_cnts, -1, (0, 255, 0), 2)
cv2.imshow("Contornos", image)

# Mostrar los dígitos y letras
print(chars)

    
