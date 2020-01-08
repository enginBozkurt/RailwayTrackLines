import numpy as np
import cv2

import matplotlib.pyplot as plt
import matplotlib.image as mping
import pickle
import glob
from util import abs_sobel_thresh, plot_figure



# Define a function that takes an image, number of x and y points,
# camera matrix and distortion coefficients
def corners_unwarp(img, nx, ny, mtx, dist):
    # Use the OpenCV undistort() function to remove distortion
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # Convert undistorted image to grayscale
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    # Search for corners in the grayscaled image
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    if ret == True:
        # If we found corners, draw them! (just for fun)
        cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
        # Choose offset from image corners to plot detected corners
        # This should be chosen to present the result at the proper aspect ratio
        # My choice of 100 pixels is not exact, but close enough for our purpose here
        offset = 100 # offset for dst points
        # Grab the image shape
        img_size = (gray.shape[1], gray.shape[0])

        # For source points I'm grabbing the outer four detected corners
        src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
        # For destination points, I'm arbitrarily choosing some points to be
        # a nice fit for displaying our warped result
        # again, not exact, but close enough for our purposes
        dst = np.float32([[offset, offset], [img_size[0]-offset, offset],
                                     [img_size[0]-offset, img_size[1]-offset],
                                     [offset, img_size[1]-offset]])
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(undist, M, img_size)

    # Return the resulting image and matrix
    return warped, M


# image = mping.imread('../camera_cal/calibration1.jpg')
# plt.imshow(image)
# plt.show()


# def cal_undistort(img, objpoints, imgpoints):
#     # Use cv2.calibrateCamera() and cv2.undistort()
#     # undist = np.copy(img)  # Delete this line
#
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)
#     img = cv2.drawChessboardCorners(img, (8, 6), corners, ret)
#
#     ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#     undist = cv2.undistort(img, mtx, dist, None, mtx)
#     return undist


# calibrate the camera
def calibrate():
    global grid_x, grid_y, img, mtx, dist
    grid_x = 9
    grid_y = 6
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((grid_y * grid_x, 3), np.float32)
    objp[:, :2] = np.mgrid[0:grid_x, 0:grid_y].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.
    # Make a list of calibration images
    images = glob.glob('camera_cal/cal*.jpg')
    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (grid_x, grid_y), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (grid_x, grid_y), corners, ret)
            write_name = 'corners_found'+str(idx)+'.jpg'
            cv2.imwrite(write_name, img)
            cv2.imshow('img', img)
            cv2.waitKey(500)
    cv2.destroyAllWindows()
    # Test  on an sample image
    # img = cv2.imread('camera_cal/calibration2.jpg')
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    img_size = (img.shape[1], img.shape[0])
    # print('image size: {}'.format(img_size))
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)


#calibrate()

# test the undistortion process on an image
def test_undistort(img):
    cv2.imshow('raw', img)
    cv2.waitKey(0)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite('output_images/test_undist.jpg',dst)
    cv2.imshow('undistorted', dst)
    cv2.waitKey(0)
    return


# save the calibration parameters for later use
def save_cal():
    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump( dist_pickle, open( "camera_cal/calib.p", "wb" ) )
    #dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)


# tets the unwarp
def test_unwarp():
    img = cv2.imread('camera_cal/calibration3.jpg')
    unwarped, M = corners_unwarp(img,grid_x, grid_y, mtx, dist)
    cv2.imshow('unwarped', unwarped)
    cv2.waitKey(0)
    return


def visualise_undistort(img ='test_images/straight_lines1.jpg'):

    #########
    # undistort the camera image

    undist = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imshow('undist', undist)
    cv2.waitKey(0)
    image_array = []
    image_titles = []
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    undist = cv2.cvtColor(undist, cv2.COLOR_BGR2RGB)
    image_array.append(img)
    image_array.append(undist)
    # image_array.append(cv2.cvtColor(combined_binary, cv2.COLOR_BGR2RGB))
    image_titles.append('img')
    image_titles.append('undist')
    plot_figure(image_array, image_titles, 1, 2, (64, 64), 'gnuplot')


# visualise_undistort()

# load the camera calibration parameters
def load_calibration():
    dist_pickle = pickle.load( open( "camera_cal/calib.p", "rb" ) )
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    return mtx, dist


# using the calibration parameters, camera matrix and distortion coefficents,
# undistort the raw input image
def undistort(img, mtx, dist):
    # img = cv2.imread('test_images/straight_lines1.jpg')
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    images = [img, undist]
    titles = ['raw image', 'undistorted image']

    # plot_figure(images,titles, 1,2)

    return undist




if __name__ == "__main__":
    calibrate()
    test_undistort(cv2.imread('camera_cal/calibration1.jpg'))