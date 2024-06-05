import csv
import math
import time
import cv2
import numpy as np
from matplotlib import pyplot as plt

from microphone import Microphone

WEBCAM_INDEX = 2
WIDTH = 1280
HEIGHT = 720
ROI_WIDTH = 1000
ROI_HEIGHT = 500

record_start_time = -1

lastROI = np.array([])

averageWindow = []
AVERAGE_WINDOW_WIDTH = 30
averageWindowPrime = []
AVERAGE_WINDOW_WIDTH_PRIME = 30
secondDerivPoints = []

output_name = 'outputVisual\\' + str(
    time.localtime().tm_mon) + "-" + str(
    time.localtime().tm_mday) + "--" + str(time.localtime().tm_hour) + "-" + str(time.localtime().tm_min) + "-" + str(
    time.localtime().tm_sec) + ".csv"

output_name_audio = "outputAudio\\" + str(
    time.localtime().tm_mon) + "-" + str(time.localtime().tm_mday) + "--" + str(time.localtime().tm_hour) + "-" + str(
    time.localtime().tm_min) + "-" + str(time.localtime().tm_sec) + ".wav"


def write_file(name, row):
    with open(name, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)


def start_capture():
    capture = cv2.VideoCapture(WEBCAM_INDEX)
    return capture


lastSmoothAvg = -1
lastDeltaSmoothAvg = -1
breathCount = 0


def process_frame(capture):
    global lastSmoothAvg
    global lastDeltaSmoothAvg
    global breathCount
    ok, frame = capture.read()
    if not ok:
        return False

    frame = cv2.resize(frame, (WIDTH, HEIGHT))

    (avgB, avgG, avgR, avg, smoothAvg) = (compute_ROI(frame, 5))

    if lastSmoothAvg == -1:
        lastSmoothAvg = smoothAvg

    averageWindowPrime.append(smoothAvg - lastSmoothAvg)
    if len(averageWindowPrime) > AVERAGE_WINDOW_WIDTH_PRIME:
        averageWindowPrime.pop(0)

    deltaSmoothAvg = np.mean(averageWindowPrime) if len(averageWindowPrime) == AVERAGE_WINDOW_WIDTH_PRIME else -1

    if lastDeltaSmoothAvg == -1:
        lastDeltaSmoothAvg = deltaSmoothAvg

    if deltaSmoothAvg > 0 > lastDeltaSmoothAvg:
        # signs are different, we breathed
        breathCount += 1
        print(f'{deltaSmoothAvg}, {lastDeltaSmoothAvg} : inhale breath {breathCount}')
        frame = cv2.rectangle(frame, (40, 40), (80, 80), (0, 255, 0), -1)

    if deltaSmoothAvg < 0 < lastDeltaSmoothAvg:
        # signs are different, we breathed
        breathCount += 1
        print(f'{deltaSmoothAvg}, {lastDeltaSmoothAvg} : exhale breath {breathCount}')
        frame = cv2.rectangle(frame, (40, 40), (80, 80), (0, 255, 0), -1)

    lastSmoothAvg = smoothAvg
    lastDeltaSmoothAvg = deltaSmoothAvg

    secondDerivPoints.append(deltaSmoothAvg)

    if record_start_time > 0:
        write_file(output_name, [time.time() - record_start_time, avgB, avgG, avgR, avg, smoothAvg, deltaSmoothAvg])

    cv2.imshow("window", frame)

    return True


def compute_ROI(frame, stroke):
    global lastROI
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #faces = face_cascade.detectMultiScale(gray, 1.4, 5)
    #for (x, y, w, h) in faces:
    x, y = 100, HEIGHT - ROI_HEIGHT
    w = ROI_WIDTH
    h = ROI_HEIGHT

    # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
    #roi_gray = gray[y:y + w, x:x + w]

    center = x + w / 2
    left = math.floor(center - ROI_WIDTH / 2)
    right = math.floor(center + ROI_WIDTH / 2)
    top = 25 + y
    bottom = top + ROI_HEIGHT

    roi = frame[top:bottom, left:right]
    avgB = np.mean(roi[:, :, 0])
    avgG = np.mean(roi[:, :, 1])
    avgR = np.mean(roi[:, :, 2])
    avg = np.mean([avgB, avgG, avgR])

    averageWindow.append(avg)
    if len(averageWindow) > AVERAGE_WINDOW_WIDTH:
        averageWindow.pop(0)

    smoothAverage = np.mean(averageWindow)

    if len(lastROI) == 0:
        lastROI = np.copy(roi)  # cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    hsbROI = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    deltaROI = np.subtract(roi[:, :, 1], lastROI[:, :, 1])

    magnification = deltaROI * 20
    lastROI[:, :, 1] += magnification
    # newROI = cv2.cvtColor(lastROI, cv2.COLOR_HSV2BGR)

    frame[top:bottom, left:right] = lastROI
    lastROI = hsbROI

    frame = cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), stroke)
    return avgB, avgG, avgR, avg, smoothAverage


def main_loop():
    global record_start_time
    capture = start_capture()
    while True:
        ok = process_frame(capture)

        key = cv2.waitKey(math.floor(1000 / 60))
        if key == ord('s'):
            record_start_time = time.time()
            mic.start()

        if not ok or key == ord('q'):
            break

    capture.release()


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mic = Microphone()
main_loop()
mic.write(output_name_audio)
mic.close()
