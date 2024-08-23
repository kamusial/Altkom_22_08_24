import cv2

# Wczytanie obrazu
image = cv2.imread('face4.jpg')

# Sprawdzenie, czy obraz został poprawnie wczytany
if image is None:
    print("Nie można otworzyć obrazu. Sprawdź ścieżkę do pliku.")
else:
    # Konwersja obrazu do skali szarości
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Wyświetlenie oryginalnego obrazu
cv2.imshow('Oryginalny obraz', image)

# Wyświetlenie obrazu w skali szarości
cv2.imshow('Obraz w skali szarości', gray_image)

# Poczekaj na naciśnięcie klawisza, aby zamknąć okna
cv2.waitKey(0)

# Zamknięcie wszystkich okien
cv2.destroyAllWindows()


