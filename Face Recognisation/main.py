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

    cap = cv2.VideoCapture(0)
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
    cap = cv2.VideoCapture(0)

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

def GenderFromImage():
    genderModel = 'deploy_gender.prototxt'
    genderProto = 'gender_net.caffemodel'
    modelMeanValues = (78.4263377603, 87.7689143744, 114.895847746)
    genderList = ('Male', 'Female')
    genderNet = cv2.dnn.readNetFromCaffe(genderModel, genderProto)

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
            face = image[start_y: end_y, start_x: end_x]
            blob = cv2.dnn.blobFromImage(image=face, scalefactor=1.0, size=(227, 227), mean=modelMeanValues,
                                         swapRB=False, crop=False)
            genderNet.setInput(blob)
            gender_preds = genderNet.forward()
            i = gender_preds[0].argmax()
            gender = genderList[i]
            genderConfidenceScore = gender_preds[0][i]
            print(genderConfidenceScore)
            label = "{}-{:.2f}%".format(gender, genderConfidenceScore * 100)
            boxColor = (30, 144, 255) if gender == "Male" else (255, 105, 180)
            cv2.rectangle(image, (start_x, start_y), (end_x, end_y), color=boxColor, thickness=2)
            cv2.putText(image, label, (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale, boxColor, 2)
    cv2.imshow("Recognisation with Neural Networks", image)
    cv2.waitKey(0)
    cv2.imwrite("FacesDetected.jpg", image)

def LiveGenderRecognistion():
    genderModel = 'deploy_gender.prototxt'
    genderProto = 'gender_net.caffemodel'
    modelMeanValues = (78.4263377603, 87.7689143744, 114.895847746)
    genderList = ('Male', 'Female')
    protoPath = "deploy.prototxt.txt"
    modelPath = "res10_300x300_ssd_iter_140000_fp16.caffemodel"

    model = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    genderNet = cv2.dnn.readNetFromCaffe(genderModel, genderProto)

    cap = cv2.VideoCapture(0)

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
                face = image[start_y: end_y, start_x: end_x]


                blob = cv2.dnn.blobFromImage(image=face, scalefactor=1.0, size=(227, 227), mean=modelMeanValues, swapRB=False, crop=False)
                genderNet.setInput(blob)
                gender_preds = genderNet.forward()
                i = gender_preds[0].argmax()
                gender = genderList[i]
                genderConfidenceScore = gender_preds[0][i]
                label ="{}-{:.2f}%".format(gender, genderConfidenceScore*100)
                boxColor = (30,144,255) if gender == "Male" else (255,105,180)
                cv2.rectangle(image, (start_x, start_y), (end_x, end_y), color=boxColor, thickness=2)
                cv2.putText(image, label, (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale, boxColor, 2)
        cv2.imshow("Live Recognisation with Neural Networks", image)



        if cv2.waitKey(1) == ord("q"):
            break
    cv2.destroyAllWindows()
    cap.release()


def LiveAgeRecognistion():
    ageModel = 'deploy_age.prototxt'
    ageProto = 'age_net.caffemodel'
    protoPath = "deploy.prototxt.txt"
    modelPath = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    modelMeanValues = (78.4263377603, 87.7689143744, 114.895847746)
    ageIntervals = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)',
                     '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
    model = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    ageNet = cv2.dnn.readNetFromCaffe(ageModel, ageProto)

    cap = cv2.VideoCapture(0)

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
                face = image[start_y: end_y, start_x: end_x]

                blob = cv2.dnn.blobFromImage(image=face, scalefactor=1.0, size=(227,227), mean=modelMeanValues, swapRB=False)
                ageNet.setInput(blob)
                agePreds = ageNet.forward()
                print("=" * 30, f"Face {i + 1} Prediction Probabilities", "=" * 30)
                for i in range(agePreds[0].shape[0]):
                    print(f"{ageIntervals[i]}: {agePreds[0, i] * 100:.2f}%")
                i = agePreds[0].argmax()
                age = ageIntervals[i]
                ageConfidenceScore = agePreds[0][i]

                label = f"Age:{age} - {ageConfidenceScore * 100:.2f}%"

                cv2.rectangle(image, (start_x, start_y), (end_x, end_y), color=(225,225,225), thickness=2)
                cv2.putText(image, label, (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color=(225,225,225),thickness= 2)


        cv2.imshow("Age Guesser", image)
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
    print('[6] Live Gender Guess - DISCLAIMER THIS IS NOT PERFECT\n[7] Live Age Guess - DISCLAIMER THIS IS NOT PERFECT')
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
    elif choice == "6":
        LiveGenderRecognistion()
    elif choice == "7":
        LiveAgeRecognistion()

if __name__ == "__main__":
    while True:
        menu()
