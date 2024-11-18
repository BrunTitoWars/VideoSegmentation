import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

background_subtractor = cv.createBackgroundSubtractorMOG2(history=2000, varThreshold=16, detectShadows=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fg_mask = background_subtractor.apply(frame)
    bg_mask = cv.bitwise_not(fg_mask)

    fg = cv.bitwise_and(frame, frame, mask=fg_mask)
    bg = cv.bitwise_and(frame, frame, mask=bg_mask)

    cv.imshow("Video Original", frame)
    cv.imshow("Plano Principal (Sem Fundo)", fg)
    cv.imshow("Plano de Fundo (Sem Principal)", bg)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
