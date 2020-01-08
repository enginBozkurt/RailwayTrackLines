
import os
import glob
import cv2
import numpy as np


# Read out all directories and files
def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles


select_ext = '*.png'
output_ext = '.png'



folder_selected = 'C:/Railway Data Science Project/Dilation/Train_Mask'
export_folder_name = 'C:/Railway Data Science Project/Dilation/Train_Mask_Dilated_New'



folder_selected_val = 'C:/Railway Data Science Project/Dilation/Val_Mask'
export_folder_name_val = 'C:/Railway Data Science Project/Dilation/Val_Mask_Dilated_New'




for filename in glob.glob(folder_selected_val + "/*.png"):
    
    # Get filname without extension
    th_fname = os.path.splitext(os.path.basename(filename))[0]
    # Load image
    img = cv2.imread(filename)
    
    
    test_img = img
    label_seg = np.zeros((test_img.shape[:2]), dtype=np.int)
    
    kernel = np.ones((3,3),np.uint8)
    
    dilation = cv2.dilate(img, kernel,iterations = 1)
    
    cv2.imwrite(os.path.join(export_folder_name_val, th_fname + output_ext), dilation)


for filename in glob.glob(folder_selected + "/*.png"):
    
    # Get filname without extension
    th_fname = os.path.splitext(os.path.basename(filename))[0]
    # Load image
    img = cv2.imread(filename)
    
    
    test_img = img
    label_seg = np.zeros((test_img.shape[:2]), dtype=np.int)
    
    kernel = np.ones((3,3),np.uint8)
    
    dilation = cv2.dilate(img, kernel,iterations = 1)
    
    cv2.imwrite(os.path.join(export_folder_name, th_fname + output_ext), dilation)
    