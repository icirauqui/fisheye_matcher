import cv2
import numpy as np

k1 = -0.139659
k2 = -0.000313999
k3 = 0.0015055
k4 = -0.000131671
fx = 717.691
fy = 718.021
cx = 734.728
cy = 552.072

import numpy as np

def initUndistortRectifyMap(K, D, R, P, size, m1type):
    map1 = np.zeros((size[1], size[0]), dtype=np.float32)
    map2 = np.zeros((size[1], size[0]), dtype=np.float32)

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    s = 0  # Skew is assumed to be 0

    k1 = D[0]
    k2 = D[1]
    k3 = D[2]
    k4 = D[3]

    fx_inv = 1.0 / fx
    fy_inv = 1.0 / fy

    for y in range(size[1]):
        for x in range(size[0]):
            x_normalized = (x - cx) * fx_inv
            y_normalized = (y - cy) * fy_inv

            r_sq = x_normalized**2 + y_normalized**2
            r = np.sqrt(r_sq)

            theta = np.arctan(r)

            if r == 0:
                theta_distorted = 1.0
            else:
                theta_distorted = theta * (1.0 + k1 * r_sq + k2 * r_sq**2 + k3 * r_sq**3 + k4 * r_sq**4)

            x_distorted = theta_distorted * x_normalized / r
            y_distorted = theta_distorted * y_normalized / r

            x_distorted_pixel = x_distorted * fx + cx
            y_distorted_pixel = y_distorted * fy + cy

            map1[y, x] = x_distorted_pixel
            map2[y, x] = y_distorted_pixel

    return map1, map2



image_i = "/home/icirauqui/workspace_phd/fisheye_matcher/images/s1_001.png";
image_o = "/home/icirauqui/workspace_phd/fisheye_matcher/images/s1_001_o.png";


# Load the distorted image
img = cv2.imread(image_i)
print(img.shape[1], img.shape[0])

# Define the camera matrix and distortion coefficients
K = np.array([[ fx, 0.0, cx],
              [0.0,  fy, cy],
              [0.0, 0.0, 1.0]])
D = np.array([k1, k2, k3, k4])
xi = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
R = np.eye(3, dtype=np.float64)
P = np.array([[img.shape[1],            0, img.shape[1]/2, 0],
              [           0, img.shape[0], img.shape[0]/2, 0],
              [           0,            0,               1, 0]], dtype=np.float64)

# Compute the distortion and rectification maps
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, R, P, img.shape[:2], cv2.CV_32F)

# Apply the distortion map to the input image
distorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

# Apply the undistortion map to the distorted image
undistorted = cv2.remap(distorted, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

# Display the input, distorted, and undistorted images side-by-side
cv2.imshow('Input', img)
cv2.imshow('Distorted', distorted)
cv2.imshow('Undistorted', undistorted)
#output = np.concatenate((img, distorted, undistorted), axis=1)
#cv2.imshow('Input | Distorted | Undistorted', output)
cv2.waitKey(0)


"""
# Undistort the fisheye image
width, height = img.shape[1], img.shape[0]
width2 = int(width*2)
height2 = int(height*2)
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (width, height), cv2.CV_16SC2)
img_undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


map1a, map1b = cv2.fisheye.initDistortRectifyMap(K, D, np.eye(3), K, (width, height), cv2.CV_16SC2)
img_undistorted2 = cv2.remap(img, map1a, map1b, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)



#img_undistorted = cv2.fisheye.undistortImage(img, K, D)

img_undistorted = cv2.resize(img_undistorted, (0,0), fx=0.5, fy=0.5)
img_undistorted2 = cv2.resize(img_undistorted2, (0,0), fx=0.5, fy=0.5)


# Concate the two images
imgs = np.hstack((img_undistorted, img_undistorted2))
cv2.imshow('Und', imgs)



cv2.waitKey(0)
"""
#cv2.destroyAllWindows()