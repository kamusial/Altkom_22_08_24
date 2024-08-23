import cv2

# Załaduj pretrenowany model Haar Cascade do wykrywania twarzy
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Wczytaj obraz, na którym chcesz wykryć twarze
image = cv2.imread('face4.jpg')

# Sprawdź, czy obraz został poprawnie wczytany
if image is None:
    print("Nie można otworzyć obrazu. Sprawdź ścieżkę do pliku.")
else:
    # Konwersja obrazu do skali szarości (wymagane przez algorytm Haar Cascade)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Wykryj twarze na obrazie
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Rysuj prostokąty wokół wykrytych twarzy
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Wyświetl wynikowy obraz z zaznaczonymi twarzami
cv2.imshow('Wykryte twarze', image)

# Poczekaj na naciśnięcie klawisza, aby zamknąć okna
cv2.waitKey(0)

# Zamknij wszystkie okna
cv2.destroyAllWindows()
