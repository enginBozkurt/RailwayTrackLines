import numpy as np
import os
import cv2

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from moviepy.editor import VideoFileClip


model_dir = 'C:/Railway Data Science Project/results/model_done.h5'
save_dir = 'C:/Railway Data Science Project/results/'

model = load_model(model_dir, custom_objects={'preprocess_input': preprocess_input})


# Binary segmentation masks

def vid_pipeline_UNET(img):
    
    img = img.astype(np.float32)
   
    frame= cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY)
    
    prcframe = cv2.resize(frame,(224, 224), interpolation = cv2.INTER_CUBIC)
    prcframe = prcframe.reshape(224, 224,1)    
    predArr = np.zeros((1, 224, 224, 1), dtype=np.float32)
    predArr[0] = prcframe
    preds_test = model.predict(predArr, verbose=0)
    preds_test_t = (preds_test > 0.5).astype(np.uint8)
    pred_frame = np.squeeze(preds_test_t[0])*255  
    
    
   
    pred_frame = cv2.resize(pred_frame,(1920, 1080), interpolation = cv2.INTER_CUBIC) 
    
   
    a = cv2.cvtColor(np.uint8( pred_frame), 4)
    
    a = cv2.cvtColor(a,cv2.COLOR_RGB2HLS)
	
    b = cv2.cvtColor(np.uint8(img), 4)
    
    output = cv2.addWeighted(a, 1, b, 1, 4, dtype=cv2.CV_32F)
    return output
    
    


output = os.path.join(save_dir, 'latest_detection_video.mp4')   
clip1 = VideoFileClip('C:/Railway Data Science Project/project_video.mp4')
white_clip = clip1.fl_image(vid_pipeline_UNET)
white_clip.write_videofile(output, audio=False)


# Colored instance segmentation masks

label_values= [(50, 13, 243), (243, 34, 13), (0, 0, 0)]


def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.

    # Arguments
        image: The one-hot format image 
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified 
        class key.
    """

    x = np.argmax(image, axis = -1)
    return x


def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.

    # Arguments
        image: single channel array where each value represents the class key.
        label_values
        
    # Returns
        Colour coded image for segmentation visualization
    """

    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x

	
def pipeline_vid(img):
    channel = 4 
    size = img.shape
    img = cv2.resize(np.float32(img),  dsize=(224, 224), interpolation = cv2.INTER_CUBIC)
    img = np.array([img])
    t = model.predict([img, img])
    output_image = reverse_one_hot(t)
    out_vis_image = colour_code_segmentation(output_image, label_values)
    
    #out_vis_image = np.squeeze(out_vis_image[0])*255 
    
    
    
    a = cv2.cvtColor(np.uint8(out_vis_image[0]), channel)
    b = cv2.cvtColor(np.uint8(img[0]), channel)
    added_image = cv2.addWeighted(a, 1, b, 1, channel, dtype=cv2.CV_32F)
    
    added_image = cv2.resize(added_image, dsize=(size[1],size[0]))

    return added_image



output = os.path.join(save_dir, 'latest_colored_UNET_detection_video.mp4')   
clip1 = VideoFileClip('C:/Railway Data Science Project/project_video.mp4')
white_clip = clip1.fl_image(pipeline_vid)
white_clip.write_videofile(output, audio=False)
