import cv2
import numpy as np
import matplotlib.pyplot as plt                     #For Plotting Confidence
from os import system
from os import listdir
from os.path import isfile, join
from pickle import dump, load           ### To store and retrieve Variables from saved file
import time                 #For FPS
import math                 # Rotating Image
import mediapipe as mp
import imutils              # Image Math operations

model = cv2.face_LBPHFaceRecognizer.create()
plot = []
criminals = []  # List Containing Criminal Names


'''Loading Criminals File with Pickle

'''
try:
    if (isfile("criminals.pickle")):  ## Checking Saved Files
        with open("criminals.pickle", "rb") as f:
            criminals = load(f)
    else:  ## Create required directories
        system("copy NUL criminals.pickle")
        system("mkdir data")
        system("mkdir model")
        system("copy /Y  NUL model\\trained.xml")
except:
    pass

#Nothing
cv2.imshow("Nothing",np.zeros([100,200], dtype=np.uint8))
cv2.destroyAllWindows()
#Nothing


def enroll_data(name,url = ""):
    global model
    pTime = 0       ## For FPS Calculation

    ## Checking if URL exists
    if url is not "":
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)  ## Capture URL
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # WebCam Capture
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if (not cap.isOpened()):
        print("URL or Webcam doesn't work !!\n")
        return
    count = 0  ## Captured Faces Count
    system("mkdir data\\\"" + name + "\"")  ## Creating Folder For Criminal Face data
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
    while (True):
        bool, img = cap.read()

        if (bool):
            #img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb_image)
            # print(result.multi_face_landmarks)
            height, width, _ = img.shape
            mask = np.zeros(img.shape[0:2], dtype=np.uint8)
            points = []
            if result.multi_face_landmarks:
                try:
                    for facial_landmarks in result.multi_face_landmarks:
                        x1, y1 = (facial_landmarks.landmark[0]).x, (facial_landmarks.landmark[0]).y
                        x2, y2 = (facial_landmarks.landmark[9]).x, (facial_landmarks.landmark[9]).y
                        rotation = 90 - math.degrees(math.atan2(-(y2 - y1), (x2 - x1)))
                        for i in range(468):
                            pt1 = facial_landmarks.landmark[i]
                            x = int(pt1.x * width)
                            y = int(pt1.y * height)
                            points.append([x, y])
                            cv2.circle(img,(x,y),1,(0,255,0))


                            #cv2.putText(img,".",(x,y),cv2.FONT_HERSHEY_DUPLEX,0.6,(0,255,0))



                        points = cv2.convexHull(np.array(points))
                        cv2.drawContours(mask, np.array([points]), -1, (255, 255, 255), -1, cv2.LINE_AA)
                        #cv2.imshow("Mask", mask)


                        res = cv2.bitwise_and(rgb_image, rgb_image, mask=mask)


                        #cv2.imshow("Res",res)
                        rect = cv2.boundingRect(np.array(points))  # returns (x,y,w,h) of the rectq
                        cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
                        # cropped = cv2.resize(cropped, (0, 0), fx=2, fy=2)
                        cropped = cv2.resize(cropped, (200,200))

                        x, y, w, h = rect[0], rect[1], rect[2], rect[3]
                        points=[]
                        cTime = time.time()
                        fps = round(1 / (cTime - pTime))
                        cropped = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
                        cropped = imutils.rotate(cropped, rotation)
                        cv2.imshow("Face", cropped)

                        pTime = cTime

                        cv2.putText(img, "fps:" + str(fps) + "   Press Q to Quit", (0, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5,(0, 255, 0))
                        cv2.putText(img, "Press C to Collect Samples: " + str(count), (0, 55), cv2.FONT_HERSHEY_DUPLEX, 0.5,(0, 255, 0))
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                except Exception as e:
                    #print(e)
                    pass
            key = cv2.waitKey(1) & 0xFF
            if (key == ord('c')):
                cv2.imwrite("./data/" + name + "/" + name + "-" + str(count) + ".jpg", cropped)
                count = count + 1
            cv2.imshow("Registering Criminal ", img)
            if (key == ord('q')):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    if(count==0):
        print("No Samples Exist!!!!")
        system("del /F/Q/S data\\\"" + name + "\"")
        return

    print("Samples Collected : ",count)
    print("---------------------------------------------")
    print("       Samples Collected Successfully")
    print("---------------------------------------------\n")
    print("\n   Training Model...................\n")
    path_train = "data/" + name + "/"
    train_data, labels = [], []
    if(name not in criminals):
        criminals.append(name)      ## Add if Criminal is New
        criminal_id = len(criminals) - 1  #Generate new unique ID
    else:
        criminal_id = criminals.index(name)   #Use Existing ID
    files = [f for f in listdir(path_train) if isfile(join(path_train, f))]  ##Finding FaceData Files
    # print(files)
    if (files is []):
        print("No Data To Train!!!!")
        return

    for file in files:
        img_path = path_train + '/' + file
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        train_data.append(np.asarray(img, dtype=np.uint8))  # Appending Face Data to train set
        labels.append(np.asarray(criminal_id, dtype=np.int32))


    try:  # Try updating Model If exists
        model.read('model/trained.xml')
        model.update(np.asarray(train_data), np.asarray(labels))
    except:
        print("Creating New Model")
        model.train(np.asarray(train_data), np.asarray(labels))  # Train New Model
    model.write('model/trained.xml')
    # system("del /Q data\*.jpg")



    print("Criminals Added : ", criminals)
    print("\n------------------------------------------------")
    print("              Training Complete")
    print("------------------------------------------------\n\n")
def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=3,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size


def detect(url = ""):
    global criminals, model, plot
    plot = []
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if url is not "" and 0:                   #if URL exists Read from URL
        cap = cv2.VideoCapture(url,cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   #Else Capture with WebCam
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if (not cap.isOpened()):
        print("URL or Webcam isn't working !!\n")
        return
    total_criminals = len(criminals)
    try:
        model.read('model/trained.xml')
    except:
        print("Model Doesn't Exist!!")
        return
    pTime =0
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=10)
    while (True):
        bool, img = cap.read()
        if (bool):
            #img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mesh = face_mesh.process(rgb_image)
            height, width, _ = img.shape
            mask = np.zeros(img.shape[0:2], dtype=np.uint8)
            points = []
            if mesh.multi_face_landmarks:
                try:
                    for facial_landmarks in mesh.multi_face_landmarks:
                        x1, y1 = (facial_landmarks.landmark[0]).x, (facial_landmarks.landmark[0]).y
                        x2, y2 = (facial_landmarks.landmark[9]).x, (facial_landmarks.landmark[9]).y
                        rotation = 90 - math.degrees(math.atan2(-(y2 - y1), (x2 - x1)))
                        for i in range(468):
                            pt1 = facial_landmarks.landmark[i]
                            x = int(pt1.x * width)
                            y = int(pt1.y * height)
                            points.append([x, y])

                        points = cv2.convexHull(np.array(points))
                        cv2.drawContours(mask, np.array([points]), -1, (255, 255, 255), -1, cv2.LINE_AA)
                        #cv2.imshow("Mask", mask)
                        res = cv2.bitwise_and(img, img, mask=mask)
                        rect = cv2.boundingRect(np.array(points))  # returns (x,y,w,h) of the rectq
                        cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
                        # cropped = cv2.resize(cropped, (0, 0), fx=2, fy=2)
                        cropped = cv2.resize(cropped, (200,200))
                        cropped = imutils.rotate(cropped , rotation)
                        face = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                        cv2.imshow("Face",face)
                        x, y, w, h = rect[0], rect[1], rect[2], rect[3]
                        result = model.predict(face)
                        confidence = float('{0:.2f}'.format(100 * (1 - (result[1]) / 300)))         #Geting Confidence
                        points=[]
                        plot.append(confidence)
                        #print(result[0]," ",confidence)
                        if (result[0] >= 0 and result[0] < total_criminals and confidence > 85):    # If ID is within label range
                            # name = str(result[0]) + " " + criminals[result[0]-1]
                            name = criminals[result[0]]
                            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            draw_text(img,str(confidence),pos=(x,y-10),font_scale=1)
                            #cv2.putText(img, str(confidence), (x, y - 10),cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255))
                            draw_text(img,name,pos=(x,y+h+20),font_scale=2)
                            #cv2.putText(img, name, (x, y + h + 50), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 0, 100))

                        else:  # If Unknown Face
                            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            draw_text(img, "Unknown ", pos=(x, y + h + 20), font_scale=2)
                            #cv2.putText(img, str(confidence), (x, y + h + 50),cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0),2)



                except Exception as e:
                    #print(e)
                    pass
            try:
                cTime = time.time()
                fps = round(1 / (cTime - pTime))  # Video FPS
                pTime = cTime
                draw_text(img, "fps:" + str(fps), pos=(0, 20), font_scale=2)
            except:
                pass
            cv2.imshow("Criminal surveillance", img)
            if (cv2.waitKey(1) & 0xFF == ord('q')):
                cap.release()
                cv2.destroyAllWindows()
                break


            #cv2.putText(img, "fps:" + str(fps) + "   Press Q to Quit", (0, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5,(0, 255, 0))


        else:
            cap.release()
            cv2.destroyAllWindows()
            break






def main():
    global plot, criminals
    while (True):
        print("-------------------------------------------------------")
        print("         Criminal Face Identification System:")
        print("-------------------------------------------------------")
        print("              1.Register Criminal Face")
        print("              2.Detect Criminal Face")
        print("              3.Delete Model")
        print("              4.Quit('q')")
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
            url = ""
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
                    #print("Average Accuracy Generated is " + str((np.min(plot) + np.max(plot)) / 2))
                    plt.plot(plot)
                    plt.ylabel("Confidence")
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
        #if opt == '4':
         ##  train_unknowns()
           # print("Training complete")

        if opt == '4':
            break


if __name__ == "__main__":
    main()
