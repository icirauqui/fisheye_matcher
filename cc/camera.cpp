#include "camera.h"




Camera::Camera(std::string path) {
  path_ = path;

  std::ifstream json_file(path_);
  nlohmann::json json_data = nlohmann::json::parse(json_file);
  nlohmann::json c1 = json_data["cam1"];

  if (c1.empty()) {
    std::cout << "Unable to load parameters from cams.json" << std::endl;
  } else {
    fx = c1["fx"];
    fy = c1["fy"];
    cx = c1["cx"];
    cy = c1["cy"];
    K_ = (cv::Mat_<float>(3, 3) << fx, 0., cx, 0., fy, cy, 0., 0., 1.);
    k1 = c1["k1"];
    k2 = c1["k2"];
    k3 = c1["k3"];
    k4 = c1["k4"];
    D_ = cv::Vec4f(k1, k2, k3, k4);
  }
}


// Compute focal lenght from K matrix
float Camera::FocalLength() {
  double lx = 2 * cx;
  double ly = 2 * cy;
  double f = (lx / (lx + ly)) * fx + (ly / (lx + ly)) * fy;
  return f;
}


// Compute camera center from K matrix
cv::Point3f Camera::CameraCenter() {
  float f = FocalLength();
  cv::Point3f c(cx, cy, f);
  return c;
}


cv::Mat Camera::K() { return K_.clone(); }


cv::Vec4f Camera::D() { return D_; }



float Camera::Cx() { return cx; }


float Camera::Cy() { return cy; }
