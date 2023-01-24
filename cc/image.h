#ifndef IMAGE_H
#define IMAGE_H

#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/viz.hpp>

class Image {

public:

  Image(std::string path);





private:


  std::vector<cv::KeyPoint> kps;

  cv::Mat desc;



}; // Image




#endif