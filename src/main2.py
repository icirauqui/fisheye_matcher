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


def main():
    img1 = cv2.imread('images/s1_001.png')
    img2 = cv2.imread('images/s1_002.png')

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    D = np.array([k1, k2, k3, k4])

    # Detect keypoints and compute descriptors
    






























if __name__ == '__main__':
    main()




