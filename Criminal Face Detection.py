import cv2
import numpy as np
import matplotlib.pyplot as plt
from os import system
from os import listdir
from os.path import isfile, join
from pickle import dump, load

face_cascade = cv2.CascadeClassifier('resources/haarcascade_frontalface_default.xml')
model = cv2.face_LBPHFaceRecognizer.create()
plot = []
criminals = []

'''Loading Criminals File with Pickle'''
try:
    with open("criminals.pickle", "rb") as f:
        criminals = load(f)
    system("mkdir data")
    system("model")
    system("cd model")
    system("copy NUL trained.xml")
    system("cd ..")
except:
    pass
#End of Pickle

def enroll_data(name):
    global model
    count = (len(criminals)-1)*100
    system("mkdir data\\" + name)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while (True):
        bool, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=6)
        if faces is ():
            # pass
            cv2.putText(img, "Show Your Face!", (0, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)
        else:
            count = count + 1
            path = 'data/' + name + "/" + name + '-' + str(count) + '.jpg'
            for x, y, w, h in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.putText(img, "Collecting Samples:" + str(count) + '/100', (0, 25), cv2.FONT_HERSHEY_DUPLEX, 1,
                            (25, 25, 25))
                face = gray[y:y + h, x:x + w]
                face = cv2.resize(face, (400, 400))
                cv2.imwrite(path, face)
        cv2.resize(img, (1024, 728))
        cv2.imshow("Resistering...", img)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break
        if count == 100:
            cap.release()
            cv2.destroyWindow("Resistering...")
            break

    print("---------------------------------------------")
    print("       Samples Collected Successfully")
    print("---------------------------------------------\n")
    print("\n   Training Model...................\n")
    path_train = "data/" + name + "/"
    train_data, labels = [], []
    files = [f for f in listdir(path_train) if isfile(join(path_train, f))]
    for i, file in enumerate(files,start=(len(criminals)-1)*100):
        img_path = path_train + '/' + files[i]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        train_data.append(np.asarray(img, dtype=np.uint8))
        labels.append(i)
    labels = np.asarray(labels, dtype=np.int32)
    # model = cv2.face_LBPHFaceRecognizer.create()
    # model.read('model/trained.xml')

    model.train(np.asarray(train_data), np.asarray(labels))
    model.write('model/trained.xml')
    # system("del /Q data\*.jpg")

    print("Criminals Added : ", criminals)
    print("\n------------------------------------------------")
    print("              Training Complete")
    print("------------------------------------------------\n\n")


def detect():
    global criminals, model
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    try:
        model.read('model/trained.xml')
    except:
        print("Model Doesn't Exist!!")
        return
    global plot
    while True:
        img = (cap.read())[1]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=6)
        if faces is ():
            cv2.putText(img, "No Criminal Detected :)", (0, 25), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0))
        else:
            for x, y, w, h in faces:
                roi = gray[y:y + h, x:x + w]
                roi = cv2.resize(roi, (400, 400))
                try:
                    result = model.predict(roi)
                    label = str(result[0]) + " " + str(round(result[1])) + " "
                    print(label)
                    if result[1] < 200:
                        confidence = float('{0:.2f}'.format(100 * (1 - (result[1]) / 300)))
                        plot.append(confidence)
                        if confidence > 86:
                            label = label + str(confidence) + '%'
                            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
                            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        else:
                            label = "Innocent :" + str(confidence)
                            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0))
                            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                except Exception as e:
                    print(e)
        cv2.resize(img,(1024,728))
        cv2.imshow("Face Detector", img)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            cv2.destroyWindow("Face Detector")
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    global plot, criminals
    while (True):
        print("-------------------------------------------------------")
        print("         Criminal Face Identification System:")
        print("-------------------------------------------------------")
        print("              1.Enroll Criminal Face")
        print("              2.Detect Criminal Face")
        print("              3.Delete Model")
        print("              4.Quit('q')")
        print("               Criminals:", criminals)
        opt = input("")
        if opt == '1':
            name = input("Enter Criminal Name:")
            criminals.append(name)
            enroll_data(name)
            with open("criminals.pickle", "wb") as f:
                dump(criminals, f)

        if opt == '2':
            detect()
            try:
                print("Highest Accuracy Generated is " + str((np.max(plot))))
                print("Average Accuracy Generated is " + str((np.min(plot) + np.max(plot)) / 2))
                plt.plot(plot)
                plt.ylabel("Accuracy")
                plt.xlabel("Runtime")
                plt.title("Confidence Plot")
                plt.show()
            except:
                pass

        if opt == '3':
            system("copy NUL model\\trained.xml")
            system("copy NUL criminals.pickle")
            system("DEL /F/Q/S data\\*.*")


            criminals = []

            print("Successfully Deleted Training Model")
        if opt == '4':
            break


if __name__ == "__main__":
    main()
