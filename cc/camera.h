#ifndef CAMERA_H
#define CAMERA_H

#include <iostream>
#include <fstream>
#include <string>
#include <math.h>

#include "../third_party/nlohmann/json.hpp"

#include <opencv2/opencv.hpp>


class Camera {

public:

  Camera(std::string path);

  float FocalLength();

  cv::Point3f CameraCenter();

  cv::Mat K();

  cv::Vec4f D();

  float Cx();

  float Cy();


private:

  std::string path_;

  float fx;
  float fy;
  float cx;
  float cy;
  cv::Mat K_ = cv::Mat_<float>(3, 3);
  float k1;
  float k2;
  float k3;
  float k4;
  cv::Vec4f D_;

  float f = 0.0;



}; // Camera




#endif