import numpy as np
import glob
import json
import math
import os
import cv2

IMAGE_PATH = "images"
SHOW_IMGS = False
UNDISTORT_IMAGES = True
UNDISTORT_IMAGE_PATH = "undistorted_images"

if UNDISTORT_IMAGES and not os.path.exists(UNDISTORT_IMAGE_PATH):
    os.mkdir(UNDISTORT_IMAGE_PATH)

# Number of rows and columns on the checkerboard
ROWS = 9
COLUMNS = 7

# The length of one box on the checkerboard (scales the coordinate to real-life)
CHECKBOARD_SQUARE_LENGTH = 1 #m

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros(((ROWS-1)*(COLUMNS-1), 3), np.float32)

objp[:,:2] = np.mgrid[0:(ROWS-1),0:(COLUMNS-1)].T.reshape(-1,2)

objp = objp[:,:] * CHECKBOARD_SQUARE_LENGTH

image_file_types = (".png", ".jpg", ".jpeg")
objpoints = [] 
imgpoints = [] 
images = []

# Add all images in the image directory
for file_type in image_file_types:
    images += glob.glob(os.path.join(IMAGE_PATH, f'*{file_type.lower()}'))

images.sort()

print("Starting calibration")

if len(images) == 0:
    raise Exception("No calibration images found, please ensure that the file path is correct")

if len(images) < 10:
    raise Exception("Please include at least 10 images or more for proper calibration")

invalid_imgs = []
file_names = []

dimensions = []

for i, path_name in enumerate(images):
    file_name = path_name[len(IMAGE_PATH):]

    print("Loading image: ", file_name)
    
    img = cv2.imread(path_name)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    dimensions.append(img.shape)

    if SHOW_IMGS:
        cv2.imshow(file_name, gray)
        cv2.waitKey(0)
    
    # Find corners of the chessboard
    ret, corners = cv2.findChessboardCorners(gray, (ROWS-1, COLUMNS-1), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    print("Board found" if ret else "Board not found!")
    
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        cv2.drawChessboardCorners(img, (ROWS-1, COLUMNS-1), corners2, ret)

        if SHOW_IMGS:
            cv2.imshow(file_name + " with corners", img)
            cv2.waitKey(0)

        file_names.append(file_name)
        
    else:
        invalid_imgs.append(file_name)

    print()

cv2.destroyAllWindows()

print(len(invalid_imgs), "/", len(images), " images were invalid", sep="")

# Get the camera matrix, distortion values, translations and euler rotations for the camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera successfully calibrated" if ret else "Failed to calibrate camera")

# Save data in JSON format
if ret:
    camera_info = {
        "camera_matrix": mtx.tolist(),
        "distortion": dist.tolist()[0],
        "image_dir_path": IMAGE_PATH,
        "image_data": []
    }

    if UNDISTORT_IMAGES:
        camera_info["undistort_image_dir_path"] = UNDISTORT_IMAGE_PATH

    for i, fname in enumerate(images):
        
        image_data = {
            "img_name": fname[len(IMAGE_PATH):].replace("\\", "").replace("/", ""),
            "tvec": np.squeeze(tvecs[i]).tolist(),
            "rvec": np.squeeze(rvecs[i]).tolist(),
            "height": dimensions[i][0],
            "width": dimensions[i][1]
        }

        if UNDISTORT_IMAGES:
            new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (dimensions[i][1], dimensions[i][0]), 1, (dimensions[i][1], dimensions[i][0]))

            mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, new_camera_mtx, (dimensions[i][1], dimensions[i][0]), 5)
            dst = cv2.remap(cv2.imread(images[i]), mapx, mapy, cv2.INTER_LINEAR)

            #x, y, w, h = roi
            #dst = dst[y:y+h, x:x+w]

            cv2.imwrite(os.path.join(UNDISTORT_IMAGE_PATH, image_data['img_name']), dst)

            image_data["img_distorted"] = image_data['img_name']

        camera_info["image_data"].append(image_data)

    with open("camera_info.json", "w") as camera_info_file:
        json.dump(camera_info, camera_info_file, indent=4)

