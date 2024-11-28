import cv2
from pyzbar.pyzbar import decode
import numpy as np
import csv
import datetime
import os
import winsound

# Function to improve the frame processing for longer barcodes
def preprocess_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    return blurred

# Function to play a beep sound
def play_beep():
    winsound.Beep(1000, 300) # For Windows

# Function to draw boundaries and recognize faces
def draw_boundary(img, classifier, scalefactor, minNeighbor, color, txt, clf):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scalefactor, minNeighbor)
    coords = []

    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        id, pred = clf.predict(gray_img[y:y + h, x:x + w])
        confidence = int(100 * (1 - pred / 300))

        if confidence > 77:
            if id == 1:
                cv2.putText(img, "person_name", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "UNKNOWN", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

        coords = [x, y, w, h]

    return coords

def recognize(img, clf, faceCascade):
    coords = draw_boundary(img, faceCascade, 1.2, 8, (255, 255, 255), "Face", clf)
    return img

# Barcode logging setup
csv_file = 'barcodes_log_final_script.csv'
with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Timestamp', 'Barcode Data'])

detected_barcodes = set()

# Load the face recognition model
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.xml")

# Open video capture
def main():
    cap = cv2.VideoCapture(1) # 1 for external camera 

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame for barcode scanning
        processed_frame = preprocess_image(frame)

        # Barcode decoding
        barcodes = decode(frame) + decode(processed_frame)
        for barcode in barcodes:
            barcode_data = barcode.data.decode('utf-8')
            barcode_type = barcode.type

            if barcode_data not in detected_barcodes:
                detected_barcodes.add(barcode_data)
                play_beep()
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow([timestamp, barcode_data])

                print(f'Detected: {barcode_data} ({barcode_type}) at {timestamp}')

            (x, y, w, h) = barcode.rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f'{barcode_data} ({barcode_type})'
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Face recognition
        frame = recognize(frame, clf, faceCascade)

        # Display the frame
        cv2.imshow('Barcode & Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
