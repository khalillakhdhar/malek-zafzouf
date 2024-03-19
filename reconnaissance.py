import cv2 
import pytesseract

#charger l'image
img = cv2.imread('image.jpeg')

#convertir l'image en niveau de gris
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#appliquer un filtre flou pour réduire le bruit
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

#utiliser les detection du contour pour trouver la plaque de la voiture
edged = cv2.Canny(blurred, 50, 150)
#trouver les contours dans l'image
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#filtrer les contours pour obtenir ceux qui peuvent être une plaque
possible_plates = []
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    if len(approx) == 4:
        possible_plates.append(approx)

#sélectionner le contour le plus grand
plate = max(possible_plates, key=cv2.contourArea)
# extraire la  région de plaque de la voiture
x, y, w, h = cv2.boundingRect(plate)
plate_img = img[y:y+h, x:x+w]
# utiliser pytesseract pour extraire le texte de la plaque
#text=pytesseract.image_to_string(plate_img)
# afficher l'image avec la plate detecté
cv2.imshow('plate détecté', plate_img)
cv2.waitKey(0)
cv2.destroyAllWindows()