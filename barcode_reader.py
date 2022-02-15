import cv2 as cv
from pyzbar.pyzbar import decode
import numpy as np
cap = cv.VideoCapture('/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_E24A993F-video-index0')
cap.set(3, 640)
cap.set(4, 480)
while True:
    ret, frame = cap.read()
    barcodes = decode(frame)
    for barcode in barcodes:
        data = barcode.data.decode('utf-8')
        pts = np.array([barcode.polygon], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv.polylines(frame, [pts], True, (0, 255 ,120), 5)
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break