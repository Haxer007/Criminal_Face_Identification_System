# Criminal Face Identification System
  A surveillance system to identify known criminals or suspects in real-time through IP Cameras.
  This project makes use of various machine learning models to identify criminal/suspect faces 
  effectively in live video stream of a lighting-controlled environment. The goal of this project
  is to help law enforcement agencies to ease their work of monitoring suspects in real-time.

######       Author: Jagadeeswar Ramanukolanu 
  
### ML Libraries used
   FaceMesh from MediaPipe Library (For Face Detection and Extraction)
   [More Info](https://google.github.io/mediapipe/solutions/face_mesh.html)
   
   Local Binary Pattern Histogram Algorithm[LBPH] from OpenCV Library (For Face Identification)
   [More Info](https://towardsdatascience.com/face-recognition-how-lbph-works-90ec258c3d6b)

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
* Face Detection - Running FaceMesh model from MediaPipe library on each Frame of video
* Training LBPH Face recognizer - Training the LBPH recognizer on stored facial images.
* Prediction— Check if the face recognizer predicts correctly for the detected face on test images 
  or video.




**Face Detection :** 
Face Detection is done using MediaPipe FaceMesh model which employs a machine learning approach
for visual object detection which is capable of processing images extremely rapidly and 
achieving high detection rates.

 MediaPipe Face Mesh model is a face geometry solution that estimates 468 3D face landmarks in
 real-time.  It employs machine learning (ML) to infer the 3D surface geometry, requiring only
 a single camera input without the need for a dedicated depth sensor. ML pipeline consists of 
 two real-time deep neural network models that work together: A detector that operates on the
 full image and computes face locations and a 3D face landmark model that operates on those 
 locations and predicts the approximate surface geometry via regression. The process can be
 easily visualized in the example below
 
 <p align="center">
  <img src="https://user-images.githubusercontent.com/73170547/129151914-4ab0915f-3719-4b16-801e-1189027b5861.png">
</p> 


**Face Identification :**
Face Identification is performed using Local Binary Pattern Histograms(LBPH)
Local Binary Pattern (LBP) is a simple yet very efficient texture operator which labels the 
pixels of an image by thresholding the neighborhood of each pixel and considers the result 
as a binary number and then to its decimal value as shown in figure below.

 <p align="center">
  <img src="https://user-images.githubusercontent.com/73170547/129152038-0fc88647-834f-4f4b-a2ff-320274689160.png">
</p> 

 
 
Now, using the image generated in the last step, we can use the Grid X and Grid Y parameters
to divide the image into multiple grids, as can be seen in the following image:

<p align="center">
  <img src="https://user-images.githubusercontent.com/73170547/129152080-e0d87293-c388-4c4c-8504-0c6bab523c1e.png">
</p> 




As we have an image in grayscale, each histogram (from each grid) will contain only 256 positions
(0~255) representing the occurrences of each pixel intensity. we need to concatenate each histogram
to create a new and bigger histogram. The final histogram represents the characteristics of the 
image original image.



**Input User Interface:**


<p align="center">
  <img src="https://user-images.githubusercontent.com/73170547/129152196-c59df14b-4ca1-4c68-be55-a7b9ebeec9f2.png">
</p> 

We can register criminal using a video URL or WebCam.
Registering criminal process goes like this when using Osama.mp4 video file:


<p align="center">
  <img src="https://user-images.githubusercontent.com/73170547/129152801-91ca0688-4b08-4d8d-b310-9cb7d6938ab4.png">
</p> 


We need to press key "C" on keyboard to capture sample at that instant.


The Identification goes like this when using Demo.mp4 video file: 


<p align="center">
  <img src="https://user-images.githubusercontent.com/73170547/129152609-599ff4ef-2dd0-4fc9-9b34-8fd4015a1794.png">
</p> 

<p align="center">
  <img src="https://user-images.githubusercontent.com/73170547/129153155-0f79a3e1-8983-4f56-8cac-cd233cd11550.png">
</p> 

<p align="center">
  <img src="https://user-images.githubusercontent.com/73170547/129153246-2fcb01df-650d-4455-a840-8e2483918726.png">
</p> 





