import cv2
import numpy as np
import matplotlib.pyplot as plt
from os import system
from os import listdir
from os.path import isfile, join
from pickle import dump, load
from tkinter import *

face_cascade = cv2.CascadeClassifier(cv2.haarcascades + 'haarcascade_frontalface_default.xml')
model = cv2.face_LBPHFaceRecognizer.create()
plot = []
criminals = []  # List Containing Criminal Names
record = [0]  # List Containing Criminal Labels
photo_count = 0
url = ""

'''Loading Criminals File with Pickle'''
try:
    if (isfile("criminals.pickle")):  ## Checking Saved Files
        with open("criminals.pickle", "rb") as f:
            criminals = load(f)
            record = load(f)
            photo_count = record[-1]
    else:  ## Create required directories
        system("copy NUL criminals.pickle")
        system("mkdir data")
        system("mkdir model")
        system("copy /Y  NUL model\\trained.xml")
except:
    pass


def get_name(id):  # Funtion Returning Criminal Name by ID
    for index, i in enumerate(record, start=0):
        if (i > id):
            return criminals[(index - 1)]


def enroll_data(name):
    global model, photo_count, url

    ## Checking if URL exists
    if url is not "":
        cap = cv2.VideoCapture(url)  ## Capture URL
        if (not cap.isOpened()):
            print("URL doesn't work !!\n Try a working url")
            return
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # WebCam Capture
    count = 0  ## Captured Faces Count
    system("mkdir data\\\"" + name + "\"")  ## Creating Directory For Criminal Facedata
    # print("mkdir data\\\"" + name +"\"")
    while (True):
        bool, img = cap.read()
        if (bool == False):
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=6)  ## Detect Faces Using Cascade
        if faces is not ():  ## if Face exists
            path = 'data/' + name + "/" + name + '-' + str(count) + '.jpg'  ## Path to store Face
            cv2.putText(img, "Collected Samples:" + str(count), (0, 25), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
            for x, y, w, h in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
                #if cv2.waitKey(1) & 0xFF == ord(' '):
                count = count + 1
                face = gray[y:y + h, x:x + w]
                face = cv2.resize(face, (200, 200))  # Resize to 200x200
                cv2.imwrite(path, face)  # Save Image to path


        cv2.resize(img, (1024, 2000))
        cv2.imshow("Enrolling Criminal ", img)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            # cv2.destroyWindow("Enrolling Criminal ")
            break
    cap.release()
    cv2.destroyAllWindows()
    record.append((count + record[-1]))
    print("---------------------------------------------")
    print("       Samples Collected Successfully")
    print("---------------------------------------------\n")
    print("\n   Training Model...................\n")
    path_train = "data/" + name + "/"
    train_data, labels = [], []
    files = [f for f in listdir(path_train) if isfile(join(path_train, f))]  ##Finding FaceData Files
    # print(files)
    if (files is []):
        print("No Data To Train!!!!")
        return

    for i, file in enumerate(files, start=0):
        img_path = path_train + '/' + files[i]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        train_data.append(np.asarray(img, dtype=np.uint8))  # Appending Face Data to
        labels.append((photo_count + i))
    labels = np.asarray(labels, dtype=np.int32)
    photo_count = photo_count + count
    # model = cv2.face_LBPHFaceRecognizer.create()
    try:  # Try updating Model If exists
        model.read('model/trained.xml')
        model.update(np.asarray(train_data), np.asarray(labels))
    except:
        model.train(np.asarray(train_data), np.asarray(labels))  # Train New Model
    model.write('model/trained.xml')
    # system("del /Q data\*.jpg")
    if name not in criminals:
        criminals.append(name)  ## Add if Criminal is New

    print("Criminals Added : ", criminals)
    print("\n------------------------------------------------")
    print("              Training Complete")
    print("------------------------------------------------\n\n")


def detect():
    global criminals, model, url
    if url is not "":
        cap = cv2.VideoCapture(url)
        if (not cap.isOpened()):
            print("URL doesn't work !!\n Try a working url")
            return
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    try:
        if (len(criminals) == 0):
            print("Please add criminals to continue!!")
            return
        model.read('model/trained.xml')
    except:
        print("Model Doesn't Exist!!")
        return
    global plot
    plot = []
    while True:
        bool, img = cap.read()
        if (bool == False):
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=8)
        if faces is not ():
            for x, y, w, h in faces:
                roi = gray[y:y + h, x:x + w]
                roi = cv2.resize(roi, (200, 200))
                try:
                    result = model.predict(roi)
                    confidence = float('{0:.2f}'.format(100 * (1 - (result[1]) / 300)))
                    if (result[0] < photo_count and result[0] >= 0 and confidence > 88):  # If ID is within label range
                        plot.append(confidence)
                        name = get_name(result[0])
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(img, str(confidence), (x, y - 10),cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
                        cv2.putText(img, name, (x, y + h + 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
                        # label=""
                        # label = str.zfill(str(result[0]), 2)
                        # label = label + " " + str(confidence)

                        #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        #if (result[1] < 58):

                        #else:
                        #    cv2.putText(img, str(confidence), (x, y - 10),
                        #                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255))

                    else:  # If Unknown Face
                        plot.append(confidence)
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                except Exception as e:
                    print(e)

        cv2.resize(img, (1024, 728))
        cv2.imshow("Face Detector", img)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            cv2.destroyWindow("Face Detector")
            break
    cap.release()
    cv2.destroyAllWindows()


def train_unknowns():
    path_train = "Unknown/"  # Unknown Facedata Path
    train_data, labels = [], []
    files = [f for f in listdir(path_train) if isfile(join(path_train, f))]
    for i, file in enumerate(files, start=0):
        img_path = path_train + '/' + files[i]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (200, 200))
        train_data.append(np.asarray(img, dtype=np.uint8))
        labels.append((-i - 1))
    labels = np.asarray(labels, dtype=np.int32)
    # print("labels=",np.asarray(labels))
    # model = cv2.face_LBPHFaceRecognizer.create()
    try:  # Try updating Model If exists
        model.read('model/trained.xml')
        model.update(np.asarray(train_data), np.asarray(labels))
    except:
        model.train(np.asarray(train_data), np.asarray(labels))  # Train New Model
    model.write('model/trained.xml')


def main():
    global plot, criminals, record, photo_count, url
    while (True):
        print("-------------------------------------------------------")
        print("         Criminal Face Identification System:")
        print("-------------------------------------------------------")
        print("              1.Enroll Criminal Face")
        print("              2.Detect Criminal Face")
        print("              3.Delete Model")
        print("              4.Install Requirements(Run this Once)")
        print("              5.Train with Unknowns(Run this Once)")
        print("              6.Quit('q')")
        print("______________________________________________________________")
        print("Criminal Data: ")
        if (len(criminals)):
            print("{ ", criminals, " }")
        else:
            print("No criminal Registered!!")

        opt = input("\nEnter Your Option:")

        if opt == '1':
            name = input("Enter Criminal Name:")  # Getting Criminal Name
            print("1.Input Criminal Footage URL")
            print("2.Use Webcam (Default)")
            opt = input("Enter Option: ")  # Input Feed
            url = ""
            if (opt == '1'):
                url = input("Enter URL:")

            enroll_data(name)
            with open("criminals.pickle", "wb") as f:
                dump(criminals, f)
                dump(record, f)

        if opt == '2':
            print("1.Input Video URL")
            print("2.Use Webcam (Default)")
            opt = input("Enter Option: ")
            url = ""
            if (opt == '1'):
                url = input("Enter URL:")
            detect()
            try:
                if (plot is not []):
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
            system("copy NUL /Y model\\trained.xml")
            system("copy NUL /Y criminals.pickle")
            system("DEL /F/Q/S data\\*.*")
            system("robocopy data data /S /move")
            system("mkdir data")

            criminals = []
            record = [0]
            photo_count = 0
            plot = [0]

            print("\n\n\nSuccessfully Deleted Training Model")
        if opt == '4':
            try:
                system("pip install opencv-contrib-python")
                system("pip install numpy")
                system("pip install pickle")
                print("\nSuccessfully Installed Requirements!\n")
            except Exception:
                print("\nFailed To Install!\n")

        if opt == '5':
            print("Training with Unknowns.........")
            train_unknowns()
            print("training complete")

        if opt == '6':
            break

'''
def GUI():
    root = Tk()
    root.title("Criminal Face Identification System")
    root.geometry("900x500")
    root['bg'] = 'grey'
    Label(root, text="Main Menu", font=('Verdana', 16), bg='grey').place(x=700, y=50)

    def enroll():
        enter = Toplevel(root)
        enter.title("Enter Details :")
        enter.geometry("400x100")
        Label(enter, text="Criminal Name :", font=(14)).grid(row=0, column=0)
        e = Entry(enter, textvariable="Enter Name Here", font=(14))
        e.grid(row=0, column=1)
        e = e.get()
        b = Button(enter, text="Confirm", height=2, width=13, font=(14), bg='#ffffff', activebackground='#000fff',
                   command=enter.destroy())
        b.pack()
        enter.mainloop()
        if e is not "":
            enroll_data(e)

    Button(root, text="Enroll Criminal", height=3, width=13, font=('verdana', 12), bg='#ffffff', command=enroll,
           activebackground='#000fff').place(x=700, y=100)
    Button(root, text="Detect Criminal", height=3, width=13, font=('verdana', 12), bg='#ffffff',
           activebackground='#000fff').place(x=700, y=200)
    Button(root, text="Clear Database", height=3, width=13, font=('verdana', 12), bg='#ffffff',
           activebackground='#000fff').place(x=700, y=300)
    Button(root, text="Quit Application", height=3, width=13, font=('verdana', 12), bg='#ffffff',
           activebackground='#000fff').place(x=700, y=400)

    root.mainloop()
    '''


if __name__ == "__main__":
    main()
