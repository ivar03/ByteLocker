import cv2
from pyzbar.pyzbar import decode
import numpy as np
import csv
import datetime
import os
import winsound

# Function to improve the frame processing for longer barcodes
def preprocess_image(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Increase the contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Apply Gaussian blur to reduce noise and enhance edges
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    return blurred

# Function to play a beep sound
def play_beep():
    #os.system('echo -e "\a"')  For Linux/Mac
    winsound.Beep(1000, 300) # For Windows

# Create/open a CSV file for storing barcode data
csv_file = 'barcodes_log.csv'
with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Timestamp', 'Barcode Data'])

cap = cv2.VideoCapture(1)

# Attempt to enable autofocus
if cap.isOpened():
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1) 

detected_barcodes = set()  # To keep track of detected barcodes to avoid duplicate logs

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame to enhance barcode features
    processed_frame = preprocess_image(frame)

    # Decode barcodes in both the original and processed frames
    barcodes = decode(frame) + decode(processed_frame)

    for barcode in barcodes:
        barcode_data = barcode.data.decode('utf-8')
        barcode_type = barcode.type

        # Check if the barcode is new to avoid duplicate logging
        if barcode_data not in detected_barcodes:
            detected_barcodes.add(barcode_data)

            play_beep()

            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Save the barcode data to the CSV file
            with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, barcode_data])

            print(f'Detected: {barcode_data} ({barcode_type}) at {timestamp}')

        # Get the bounding box coordinates for the barcode
        (x, y, w, h) = barcode.rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the data and type on the frame
        text = f'{barcode_data} ({barcode_type})'
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the original frame with the detected barcode
    cv2.imshow('Barcode Scanner', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
