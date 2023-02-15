/*
This function takes three arguments:

K: the camera intrinsic matrix
distCoeffs: the camera distortion coefficients
pt: the 2D point to transform
The function first converts the 2D point to a 3D point in homogeneous coordinates. It then undistorts the 2D point using the distortion coefficients, and converts the undistorted 2D point to 3D coordinates using the Kannala-Brandt model. Finally, the function scales the 3D coordinates by the distance from the camera to obtain the final 3D point.

Note that this function assumes that the camera is calibrated using the Kannala-Brandt model and that the distortion coefficients are provided. If you are using a different camera model or calibration method, you may need to modify the function accordingly.
*/

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

cv::Mat transformPointTo3D(cv::Mat K, cv::Mat distCoeffs, cv::Point2d pt)
{
    // Convert the 2D point to a 3D point in homogeneous coordinates
    cv::Mat ptHomogeneous = cv::Mat::ones(3, 1, CV_64F);
    ptHomogeneous.at<double>(0, 0) = pt.x;
    ptHomogeneous.at<double>(1, 0) = pt.y;

    // Undistort the 2D point using the distortion coefficients
    cv::Mat undistortedPt;
    cv::undistortPoints(ptHomogeneous, undistortedPt, K, distCoeffs);

    // Convert the undistorted 2D point to 3D coordinates using the Kannala-Brandt model
    cv::Mat pt3D = cv::Mat::zeros(3, 1, CV_64F);
    double r = cv::norm(undistortedPt);
    double theta = atan(r);
    double phi = atan2(undistortedPt.at<double>(1, 0), undistortedPt.at<double>(0, 0));
    pt3D.at<double>(0, 0) = sin(theta) * cos(phi);
    pt3D.at<double>(1, 0) = sin(theta) * sin(phi);
    pt3D.at<double>(2, 0) = cos(theta);

    // Scale the 3D coordinates by the distance from the camera
    cv::Mat invK = K.inv();
    cv::Mat ray = invK * ptHomogeneous;
    double distance = pt3D.dot(ray);
    pt3D *= distance;

    return pt3D;
}




/*
This function takes the same five arguments as the previous example:

K: the camera intrinsic matrix
distCoeffs: the camera distortion coefficients for the Kannala-Brandt model
R: the rotation matrix from world coordinates to camera coordinates
t: the translation vector from world coordinates to camera coordinates
pt3D: the 3D point to project
The function first transforms the 3D point to camera coordinates using the rotation matrix and translation vector. It then projects the point onto the image plane using the Kannala-Brandt model, taking into account the distortion coefficients. Finally, the function converts the point to pixel coordinates and returns it as a cv::Point2d.

Note that the Kannala-Brandt model is a more complex model than the standard pinhole camera model, and it requires five distortion coefficients to be specified. The function provided here assumes that the distortion coefficients are provided in the distCoeffs argument as a 1x5 matrix. If your camera uses a different number of distortion coefficients, or a different model, you may need to modify the function accordingly.
*/

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

cv::Point2d projectPointTo2D(cv::Mat K, cv::Mat distCoeffs, cv::Mat R, cv::Mat t, cv::Mat pt3D)
{
    // Transform the 3D point to camera coordinates
    cv::Mat pt3DCam = R * pt3D + t;

    // Project the 3D point onto the image plane using the Kannala-Brandt model
    double x = pt3DCam.at<double>(0, 0) / pt3DCam.at<double>(2, 0);
    double y = pt3DCam.at<double>(1, 0) / pt3DCam.at<double>(2, 0);
    double r = sqrt(x * x + y * y);
    double theta = atan(r);
    double phi = atan2(y, x);
    double a = distCoeffs.at<double>(0, 0);
    double b = distCoeffs.at<double>(0, 1);
    double c = distCoeffs.at<double>(0, 2);
    double d = distCoeffs.at<double>(0, 3);
    double e = distCoeffs.at<double>(0, 4);
    double rd = theta + a * theta * theta + b * theta * theta * theta * theta
                + c * theta * theta * theta * theta * theta * theta
                + d * theta * theta * theta * theta * theta * theta * theta * theta
                + e * theta * theta * theta * theta * theta * theta * theta * theta * theta * theta;
    double u = K.at<double>(0, 0) * rd * cos(phi) + K.at<double>(0, 2);
    double v = K.at<double>(1, 1) * rd * sin(phi) + K.at<double>(1, 2);

    // Convert the point to pixel coordinates
    cv::Point2d ptPixel(u, v);

    return ptPixel;
}
