import cv2
import numpy as np
import matplotlib.pyplot as plt
from os import system
from os import listdir
from os.path import isfile, join
from pickle import dump, load
#from tkinter import *

face_cascade = cv2.CascadeClassifier(cv2.haarcascades + 'haarcascade_frontalface_default.xml')
model = cv2.face_LBPHFaceRecognizer.create()
plot = []          # Plot Variable storing Confidence plot
criminals = []  # List Containing Criminal Names


'''Loading Criminals File with Pickle'''
try:
    if (isfile("criminals.pickle")):          ## Checking Saved Files
        with open("criminals.pickle", "rb") as f:
            criminals = load(f)
    else:                                   ## Create required directories
        system("copy NUL criminals.pickle")
        system("mkdir data")
        system("mkdir model")
        system("copy /Y NUL model\\trained.xml")
except:
    pass



def enroll_data(name,url=""):
    global model

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
        cv2.putText(img, "Press 'q' to Quit!", (0, 20), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0))
        #img = cv2.resize(img, (0, 0), fx=0.8, fy=0.8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6)  ## Detect Faces Using Cascade
        if faces is not ():  ## if Face exists
            path = 'data/' + name + "/" + name + '-' + str(count) + '.jpg'  ## Path to store Face
            cv2.putText(img, "Taking Samples: " + str(count), (0, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0))
            for x, y, w, h in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
                #if cv2.waitKey(1) & 0xFF == ord(' '):
                count = count + 1
                face = gray[y:y + h, x:x + w]
                face = cv2.resize(face, (200,200))  # Resize image
                cv2.imwrite(path, face)  # Save Image to path


        img = cv2.resize(img, (1024,700))
        cv2.imshow("Enrolling Criminal ", img)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            # cv2.destroyWindow("Enrolling Criminal ")
            break
    cap.release()
    cv2.destroyAllWindows()
    print("---------------------------------------------")
    print("       Samples Collected Successfully")
    print("---------------------------------------------\n")
    print("\n   Training Model...................\n")
    path_train = "data/" + name + "/"
    train_data, labels = [], []
    criminal_id = 0 
    if(name not in criminals):
        criminals.append(name)      ## Add if Criminal is New
        criminal_id = len(criminals)  #Generate new unique ID
    else:
        criminal_id = criminals.index(name)+1   #Use Existing ID
    files = [f for f in listdir(path_train) if isfile(join(path_train, f))]  ##Finding FaceData Files
    # print(files)
    if (files is []):
        print("No Data To Train!!!!")
        return

    for i, file in enumerate(files, start=0):
        img_path = path_train + '/' + files[i]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        train_data.append(np.asarray(img, dtype=np.uint8))  # Appending Face Data to train set
        labels.append(criminal_id)
    labels = np.asarray(labels, dtype=np.int32)
    # model = cv2.face_LBPHFaceRecognizer.create()
    try:  # Try updating Model If exists
        model.read('model/trained.xml')                             # Update Previous Model
        model.update(np.asarray(train_data), np.asarray(labels))
    except:
        print("Creating New Model")
        model.train(np.asarray(train_data), np.asarray(labels))          # Train New Model
    model.write('model/trained.xml')
    # system("del /Q data\*.jpg")



    print("Criminals Added : ", criminals)
    print("\n------------------------------------------------")
    print("              Training Complete")
    print("------------------------------------------------\n\n")


def detect(url=""):
    global criminals, model
    if url is not "":                   #if URL exists Read from URL
        cap = cv2.VideoCapture(url)
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   #Else Capture with WebCam
    if (not cap.isOpened()):
        print("URL or Webcam isn't working !!\n")
        return
    total_criminals = len(criminals)
    try:
        model.read('model/trained.xml')         # Read Model
    except:
        print("Model Doesn't Exist!!")
        return
    global plot
    plot = []
    while True:
        bool, img = cap.read()
        if (bool == False):
            break
        #img = cv2.resize(img, (0, 0), fx=0.7, fy=0.7)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.putText(img, "Press 'q' to Quit!", (0, 20), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0))
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6)
        if faces is not ():
            for x, y, w, h in faces:
                roi = gray[y:y + h, x:x + w]
                roi = cv2.resize(roi, (200,200)) # resizing Image
                try:
                    result = model.predict(roi)
                    confidence = float('{0:.2f}'.format(100 * (1 - (result[1]) / 300)))
                    plot.append(confidence)
                    if (result[0] > 0 and result[0] <= total_criminals and confidence > 85):  # If ID is within label range
                        name = criminals[result[0] - 1]         #Getting Name from ID (result[0])
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(img, str(confidence), (x, y - 10),cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
                        cv2.putText(img,name, (x, y + h + 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

                    else:  # If Unknown Face
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(img,str(confidence), (x, y + h + 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255,0), 2)

                except Exception as e:
                    print(e)

        img = cv2.resize(img, (1024,700))
        cv2.imshow("Face Detector", img)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            cv2.destroyWindow("Face Detector")
            break
    cap.release()
    cv2.destroyAllWindows()


def train_unknowns():
    path_train = "Unknown Dataset/"  # Unknown Facedata Path
    train_data, labels = [], []
    files = [f for f in listdir(path_train) if isfile(join(path_train, f))]
    for i, file in enumerate(files, start=0):
        img_path = path_train + '/' + files[i]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (200, 200))
        train_data.append(np.asarray(img, dtype=np.uint8))
        labels.append((999))
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
    global plot, criminals
    while (True):
        print("-------------------------------------------------------")
        print("         Criminal Face Identification System:")
        print("-------------------------------------------------------")
        print("              1.Register Criminal Face")
        print("              2.Detect Criminal Face")
        print("              3.Delete Model")
        print("              4.Train with Unknowns(Run this Once)")
        print("              5.Quit('q')")
        print("______________________________________________________________")
        print("Criminal Data: ")
        if (len(criminals)):
            print("Total Registered Criminals: ",len(criminals))
            print("{ ", criminals, " }")
        else:
            print("No criminal Registered!!")

        opt = input("\nEnter Your Option:")

        if opt == '1':
            name = input("Enter Criminal Name:")  # Getting Criminal Name
            print("\nRegister Criminal From?\n1.Input Criminal Footage URL")
            print("2.Use Webcam (Default)")
            opt = input("Enter Option: ")  # Input Feed
            if (opt == '1'):
                url = input("Enter URL:")
            enroll_data(name,url)
            with open("criminals.pickle", "wb") as f:
                dump(criminals, f)

        if opt == '2':
            if (len(criminals) == 0):
                input("Please add criminals to Continue!!\n")
                continue
            print("\nDetect Criminal From?\n1.Input Video URL")
            print("2.Use Webcam (Default)")
            opt = input("Enter Option: ")
            url = ""
            if (opt == '1'):
                url = input("Enter URL:")
            detect(url)
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
            plot = [0]

            print("\n\n\nSuccessfully Deleted Training Model")
        if opt == '4':
            print("Training with Unknowns.........")
            train_unknowns()
            print("training complete")

        if opt == '5':
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
