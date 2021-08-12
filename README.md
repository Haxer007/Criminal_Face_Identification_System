# Criminal Face Identification System
## Introduction:
**What is Face Identification System?**

A Face Identificatiion System is a technology capable of matching a human face from a digital
image or a video frame against a database of faces.

**Usage:**
This Face Identification System can be trained to identify criminal faces which can be
implemented into the Security System to detect the criminal faces.This system can be used in
ATM cameras and Street cameras to detect the criminals, if the face of the criminal or suspect
is in the database of Police Department.

The working of code can be boiled down to three main processes namely :
 - **Training the LBPH model** 
 - **Detecting Faces in Video Feed**
 - **Recognizing faces via trained LBPH model**  



**Face Detection :** 
Face Detection is done using MediaPipe FaceMesh model which employs a machine learning approach for visual object detection which is capable of processing images extremely rapidly and achieving high detection rates.
**Face Identification :**
Face Identification is performed using Local Binary Pattern Histograms(LBPH)
The entire process of code can be divided into these three processes :
*Face Detection - Running FaceMesh model from MediaPipe library on each Frame of video
Training LBPH Face recognizer - Training the LBPH recognizer on stored facial images.
Prediction – Check if the face recognizer predicts correctly for the detected face on test images or video.*
![image](https://user-images.githubusercontent.com/73170547/129151381-09a34e60-0a74-411b-9c39-77907f81dfc4.png)

