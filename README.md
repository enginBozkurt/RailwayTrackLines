# RailwayTrackLines
Railway Tracking Lines Project

## Introduction
In this project, the main goal is to detecting railway tracks from the camera images.

## Getting Started

- We are going to use a set of OpenCV routines in order to apply correction for Camera calibration. **CameraCalibration.ipynb**

- For the dilation process of the binary masks, you can use  **DilateMask.py**  script

- For splitting dataset into releveant directories,  you can use **SplitData.py**

- For converting colored labeles into binary segmentation labels, you can use  **RGB2BinaryMask.py**

## Solution Approaches

I followed three approaches to implement the solution:

- Using colored instance masks, I implemented the UNET model for semantic segmentation to detect railway tracks.  **ColormaskUnet.ipynb**

- RGB (3 channel) labels are converted to  binary segmentation labels and split them into single classes (single - channel labels).
  The binary segmentation uses 255 (pixel color value) to represent the tracks and 0 for the rest.
  We are getting three classes of labels(background, boundary, and object classes) after the transform process.
  We are just using the instances (samples) of object classes as mask labels. 
  We are feeding these labels to the following network models:
   - Unet     ---->   **UnetSegModel.ipynb**
   -  Enet    ----->     **ENet.ipynb**

- The dilation technique (a kind of Morphological Transformation technique)  is applied to binary masks, and using these dilated masks, I implemented Unet model  for semantic segmentation.  -->  **UnetSegDilation.ipynb**

## Final notes
**Draw_Rails_Video.py**   script draws detected railway tracks into the video.

## Detection Results on video

![video1](https://user-images.githubusercontent.com/30608533/72059125-78dd7b00-32e2-11ea-9706-230731fa62c0.jpg)

![video2](https://user-images.githubusercontent.com/30608533/72059145-83981000-32e2-11ea-8254-c4700c6b6cfe.jpg)
