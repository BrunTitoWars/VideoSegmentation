import cv2 as cv
import mediapipe as mp
import numpy as np

mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao abrir a webcam!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv.resize(frame, (640, 480))

    frame_rgb = cv.cvtColor(frame_resized, cv.COLOR_BGR2RGB)

    results = selfie_segmentation.process(frame_rgb)

    fg_mask = results.segmentation_mask

    _, binary_mask = cv.threshold(fg_mask, 0.5, 255, cv.THRESH_BINARY)
    binary_mask = binary_mask.astype(np.uint8)

    foreground = cv.bitwise_and(frame_resized, frame_resized, mask=binary_mask)

    cv.imshow("Original Frame", frame_resized)
    cv.imshow("Foreground Mask", binary_mask)
    cv.imshow("Isolated Foreground", foreground)

    if cv.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
