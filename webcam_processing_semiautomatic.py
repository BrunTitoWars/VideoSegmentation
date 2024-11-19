import cv2 as cv
import numpy as np

background_subtractor = cv.createBackgroundSubtractorMOG2(history=2000, varThreshold=16, detectShadows=True)

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao abrir a webcam!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv.resize(frame, (640, 480))

    fg_mask = background_subtractor.apply(frame_resized)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_OPEN, kernel)

    contours, _ = cv.findContours(fg_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv.contourArea(contour) > 500:
            x, y, w, h = cv.boundingRect(contour)
            cv.rectangle(frame_resized, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv.imshow("Webcam", frame_resized)
    cv.imshow("Máscara", fg_mask)

    if cv.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
