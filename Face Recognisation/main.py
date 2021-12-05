import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()


def RecogniseFromImage():
    filePath = filedialog.askopenfilename()
    print(filePath)
    image = cv2.imread(filePath)
    imageColourConversion = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faceCascase = cv2.CascadeClassifier("haarcascade_frontalface.default.xml")
    faces = faceCascase.detectMultiScale(imageColourConversion)
    for x, y, width, height in faces:
        cv2.rectangle(image, (x, y), (x + width, y + height), color=(255, 255, 255), thickness=2)
    cv2.imwrite("FacesDetected.jpg", image)


def RecogniseFromWebcam():
    cap = cv2.VideoCapture(1)
    faceCascase = cv2.CascadeClassifier("haarcascade_frontalface.default.xml")

    while True:
        _, image = cap.read()
        imageColourConversion = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = faceCascase.detectMultiScale(imageColourConversion)
        for x, y, width, height in faces:
            cv2.rectangle(image, (x, y), (x + width, y + height), color=(255, 255, 255), thickness=2)
        cv2.imshow("Live recognisation with Cascase", image)
        if cv2.waitKey(1) == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def RecognisingFacesWithNeuralNetworks():
    filePath = filedialog.askopenfilename()
    print(filePath)
    protoPath = "deploy.prototxt.txt"
    modelPath = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    model = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    image = cv2.imread(filePath)
    h, w = image.shape[:2]
    resizedImage = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    model.setInput(resizedImage)
    output = np.squeeze(model.forward())
    fontScale = 1.0
    for i in range(0, output.shape[0]):
        confidence = output[i, 2]
        if (confidence > 0.5):
            box = output[i, 3:7] * np.array([w, h, w, h])
            start_x, start_y, end_x, end_y = box.astype(np.int)
            cv2.rectangle(image, (start_x, start_y), (end_x, end_y), color=(255, 255, 255), thickness=2)
            cv2.putText(image, f"{confidence * 100:.2f}%", (start_x, start_y - 5), cv2.FONT_HERSHEY_SIMPLEX, fontScale,
                        (255, 255, 255), 2)
    cv2.imshow("Recognisation with Neural Networks", image)
    cv2.waitKey(0)
    cv2.imwrite("FacesDetected.jpg", image)


def RecognisingWebcamWithNeuralNetworks(options=False):
    protoPath = "deploy.prototxt.txt"
    modelPath = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    model = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    cap = cv2.VideoCapture(1)

    while True:
        _, image = cap.read()
        h, w = image.shape[:2]
        kWidth = (w // 7) | 1
        kHeight = (h // 7) | 1
        resizedImage = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
        model.setInput(resizedImage)
        output = np.squeeze(model.forward())
        fontScale = 1.0
        for i in range(0, output.shape[0]):
            confidence = output[i, 2]
            if confidence > 0.5:
                box = output[i, 3:7] * np.array([w, h, w, h])
                start_x, start_y, end_x, end_y = box.astype(np.int)
                if options == True:
                    face = image[start_y: end_y, start_x: end_x]
                    face = cv2.GaussianBlur(face, (kWidth, kHeight), 0)
                    image[start_y: end_y, start_x: end_x] = face
                cv2.rectangle(image, (start_x, start_y), (end_x, end_y), color=(255, 255, 255), thickness=2)
                cv2.putText(image, f"{confidence * 100:.2f}%", (start_x, start_y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale, (255, 255, 255), 2)
        cv2.imshow("Live Recognisation with Neural Networks", image)
        if cv2.waitKey(1) == ord("q"):
            break
    cv2.destroyAllWindows()
    cap.release()


def menu():
    print("Please Select Menu Item")
    print('[1] Image Recognisation\n[2] Camera Recognisation')
    print(
        '[3] Neural Network Image Recognisation\n[4] Neural Network Camera recognisation No Blur\n[5] Neural Network '
        'Camera Recognisation With Blur')
    choice = input("Options: ")
    if choice == "1":
        RecogniseFromImage()
    elif choice == "2":
        RecogniseFromWebcam()
    elif choice == "3":
        RecognisingFacesWithNeuralNetworks()
    elif choice == "4":
        RecognisingWebcamWithNeuralNetworks()
    elif choice == "5":
        RecognisingWebcamWithNeuralNetworks(True)


if __name__ == "__main__":

    while True:
        menu()
