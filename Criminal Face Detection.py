import cv2
import numpy as np
import matplotlib.pyplot as plt
from os import system
from os import listdir
from os.path import isfile, join
from pickle import dump, load

face_cascade = cv2.CascadeClassifier(cv2.haarcascades + 'haarcascade_frontalface_default.xml')
model = cv2.face_LBPHFaceRecognizer.create()
plot = []
criminals = []
record =[0]
photo_count = 0
url = ""

'''Loading Criminals File with Pickle'''
try:
    if (isfile("criminals.pickle")):
        with open("criminals.pickle", "rb") as f:
            criminals = load(f)
            record = load(f)
            photo_count=record[-1]
    else:
        system("copy NUL criminals.pickle")
        system("mkdir data")
        system("mkdir model")
        system("copy /Y  NUL model\\trained.xml")
except:
    pass

def get_name(id):
    for index,i in enumerate(record,start=0):
        if(i>id):
            return criminals[(index-1)]

# End of Pickle

def enroll_data(name):
    global model,photo_count,url
    if url is not "":
        cap = cv2.VideoCapture(url)
        if (not cap.isOpened()):
            print("URL doesn't work !!\n Try a working url")
            return
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if name not in criminals:
        criminals.append(name)
    count = 0
    system("mkdir data\\" + name)
    while (True):
        img = cap.read()[1]
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except:
            cap.release()
            cv2.destroyAllWindows()
            return
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=6)
        if faces is not ():
            path = 'data/' + name + "/" + name + '-' + str(count) + '.jpg'
            cv2.putText(img, "Collected Samples:" + str(count), (0, 25), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
            for x, y, w, h in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
                #cv2.putText(img, "Collected Samples:" + str(count), (0, 25), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
                if cv2.waitKey(1) & 0xFF == ord(' '):
                    count = count + 1
                    face = gray[y:y + h, x:x + w]
                    face = cv2.resize(face, (200, 200))
                    cv2.imwrite(path, face)

        cv2.resize(img, (1024, 2000))
        cv2.imshow("Enrolling Criminal ", img)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            cv2.destroyWindow("Enrolling Criminal ")
            break
    cap.release()
    if(name in criminals):
        record[criminals.index(name)]+=count
    else:
        record.append((count+record[-1]))
    print("---------------------------------------------")
    print("       Samples Collected Successfully")
    print("---------------------------------------------\n")
    print("\n   Training Model...................\n")
    path_train = "data/" + name + "/"
    train_data, labels = [], []
    files = [f for f in listdir(path_train) if isfile(join(path_train, f))]
    #print(files)
    for i, file in enumerate(files, start = 0):
        img_path = path_train + '/' + files[i]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        train_data.append(np.asarray(img, dtype=np.uint8))
        labels.append((photo_count+i))
    labels = np.asarray(labels, dtype=np.int32)
    photo_count=photo_count + count
    # model = cv2.face_LBPHFaceRecognizer.create()
    try:                                                        #Try updating Model If exists
        model.read('model/trained.xml')
        model.update(np.asarray(train_data), np.asarray(labels))
    except:
        model.train(np.asarray(train_data), np.asarray(labels)) #Train New Model
    model.write('model/trained.xml')
    # system("del /Q data\*.jpg")

    print("Criminals Added : ", criminals)
    print("\n------------------------------------------------")
    print("              Training Complete")
    print("------------------------------------------------\n\n")


def detect():
    global criminals, model,url
    if url is not "":
        cap= cv2.VideoCapture(url)
        if(not cap.isOpened()):
            print("URL doesn't work !!\n Try a working url")
            return
    else:
        cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    try:
        if(len(criminals)==0):
            print("Please add criminals to continue!!")
            return
        model.read('model/trained.xml')
    except:
        print("Model Doesn't Exist!!")
        return
    global plot
    while True:
        img = (cap.read())[1]
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except:
            cap.release()
            cv2.destroyAllWindows()
            return
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=6)
        if faces is not ():
            for x, y, w, h in faces:
                roi = gray[y:y + h, x:x + w]
                roi = cv2.resize(roi, (200, 200))
                try:
                    result = model.predict(roi)
                    print(result)
                    confidence = float('{0:.2f}'.format(100 * (1 - (result[1]) / 300)))
                    if(result[0]<photo_count):
                        if (result[0] >= 0):
                            plot.append(confidence)
                            label = str.zfill(str(result[0]), 2)
                            label = label + " " + str(confidence)
                            name = get_name(result[0])
                            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            if (result[1] < 58):
                                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
                                cv2.putText(img, name, (x, y + h + 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
                            else:
                                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0))
                        else:
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
    path_train = "Unknown Dataset/"                                                 # Train Path for Unknown Faces
    train_data, labels = [], []
    files = [f for f in listdir(path_train) if isfile(join(path_train, f))]
    for i, file in enumerate(files, start=0):
        img_path = path_train + '/' + files[i]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img= cv2.resize(img,(200,200))
        train_data.append(np.asarray(img, dtype=np.uint8))
        labels.append((-i-1))
    labels = np.asarray(labels, dtype=np.int32)
    #print("labels=",np.asarray(labels))
    # model = cv2.face_LBPHFaceRecognizer.create()
    try:                                                                     # Try updating Model If exists
        model.read('model/trained.xml')
        model.update(np.asarray(train_data), np.asarray(labels))
    except:
        model.train(np.asarray(train_data), np.asarray(labels))              # Train New Model
    model.write('model/trained.xml')



def main():
    global plot, criminals,record,photo_count,url
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
        #print("record=",record,", Photo count=",photo_count)
        if(len(criminals)):
            print("{ ",criminals," }")
        else:
            print("No criminal Registered!!")
        print("Records: ",record,"   ",photo_count)

        opt = input("\n\t\t\tEnter Your Option:")


        if opt == '1':
            name = input("Enter Criminal Name:")
            print("1.Input Criminal Footage URL")
            print("2.Use Webcam (Default)")
            opt = input("Enter Option: ")
            url = ""
            if (opt == '1'):
                url = input("Enter URL:")
            enroll_data(name)
            with open("criminals.pickle", "wb") as f:
                dump(criminals, f)
                dump(record,f)


        if opt == '2':
            print("1.Input Video URL")
            print("2.Use Webcam (Default)")
            opt=input("Enter Option: ")
            url=""
            if(opt=='1'):
                url = input("Enter URL:" )
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
            system("copy NUL /Y model\\trained.xml")    # Clear Model File  
            system("copy NUL /Y criminals.pickle")      # Clear Criminal Data
            system("DEL /F/Q/S data\\*.*")              
            system("rmdir /S /Q  New\\")
            system("mkdir data")



            criminals = []
            record= [0]
            photo_count= 0
            plot=[0]
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


if __name__ == "__main__":
    main()
