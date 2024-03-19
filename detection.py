import cv2

# Charger l'image
img = cv2.imread('image.jpeg')

# Convertir l'image en niveaux de gris
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Appliquer un filtre flou pour réduire le bruit
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Utiliser la détection du contour pour trouver la plaque de la voiture
edged = cv2.Canny(blurred, 50, 150)

# Trouver les contours dans l'image
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filtrer les contours pour obtenir ceux qui peuvent être une plaque
possible_plates = []
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    if len(approx) == 4:
        possible_plates.append(approx)

# Sélectionner le contour le plus grand
plate = max(possible_plates, key=cv2.contourArea)

# Extraire la région de plaque de la voiture
x, y, w, h = cv2.boundingRect(plate)
plate_img = img[y:y+h, x:x+w]

# Afficher l'image avec la plaque détectée
cv2.imshow('Plaque détectée', plate_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
