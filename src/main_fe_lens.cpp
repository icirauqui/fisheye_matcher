#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp> // include the viz module

#include <fstream>


#include "fe_lens/fe_lens.hpp"
#include "matcher/matcher.hpp"


#include "src/ang_matcher/ang_matcher.h"


void FullLens() {
    // Set the camera intrinsics
  FisheyeLens lens(717.2104, 717.4816, 735.3566, 552.7982, 
                   -0.1389272, -0.001239606, 0.0009125824, -0.00004071615);

  // Define max angles for semi-sphere
  double theta_max = 60 * M_PI / 180;  // 60 degrees
  double phi_max = 2 * M_PI;  // 360 degrees
  std::cout << "Theta max: " << theta_max << std::endl;
  std::cout << "Phi max: " << phi_max << std::endl;

  // Generate 3D points over the semi-sphere
  std::vector<std::vector<double>> coords3d;
  coords3d.push_back(std::vector<double>({0.0, 0.0}));  // Center
  double resolution = 0.05;
  for (double theta = resolution; theta < theta_max; theta += resolution) {
    for (double phi = 0.0; phi < phi_max; phi += resolution) {
      coords3d.push_back(std::vector<double>({theta, phi}));
    }
  }

  // Project the semi-sphere onto the image plane
  std::vector<cv::Point2f> coords2d;
  //for (auto coord : coords3d) {
  for (unsigned int i=0; i<coords3d.size(); i++) {
    double theta = coords3d[i][0];
    double phi = coords3d[i][1];
    cv::Point2f coord = lens.Compute2D(theta, phi, true);
    coords2d.push_back(coord);
  }

  // Project back the image points onto the semi-sphere
  std::vector<std::vector<double>> coords3d_reconstr;
  for (auto coord : coords2d) {
    double x = coord.x;
    double y = coord.y;
    coords3d_reconstr.push_back(lens.Compute3D(x, y, true));
    //if (it == 3)
    //  break;
  }

  // Compute the error 
  double error = lens.ComputeError(coords3d, coords3d_reconstr);
  std::cout << "Global error = " << error << std::endl;


  // - - - - Visualization - - - - - - - - - - - - - - - - - -

  float scale = 1;
  float offset = 0.75;

  // Original Lens
  std::vector<cv::Point3f> points_lens;
  std::vector<cv::Vec3b> colors_lens;
  for (auto coord: coords3d) {
    double x = scale * sin(coord[0]) * cos(coord[1]);
    double y = scale * sin(coord[0]) * sin(coord[1]);
    double z = offset + scale * cos(coord[0]);
    points_lens.push_back(cv::Point3f(x, y, z));
    colors_lens.push_back(cv::Vec3b(255, 255, 255));
  }
  
  // Projected image
  std::vector<cv::Point3f> points_image;
  std::vector<cv::Vec3b> colors_image;
  for (auto coord: coords2d) {
    double x = scale * coord.x;
    double y = scale * coord.y;
    double z = -offset;
    points_image.push_back(cv::Point3f(x, y, z));
    colors_image.push_back(cv::Vec3b(0, 0, 255));
  }  

  // Reconstructed lens
  std::vector<cv::Point3f> points_lens_reconstr;
  std::vector<cv::Vec3b> colors_lens_reconstr;
  for (auto coord: coords3d_reconstr) {
    double x = scale * sin(coord[0]) * cos(coord[1]);
    double y = scale * sin(coord[0]) * sin(coord[1]);
    double z = 2*offset + scale * cos(coord[0]);
    points_lens_reconstr.push_back(cv::Point3f(x, y, z));
    colors_lens_reconstr.push_back(cv::Vec3b(0, 255, 0));
  }


  Visualizer vis;
  vis.AddCloud(points_lens, colors_lens);
  vis.AddCloud(points_image, colors_image);
  vis.AddCloud(points_lens_reconstr, colors_lens_reconstr);
  vis.Render();
}



void ImgMethod() {
    // Set the camera intrinsics
  FisheyeLens lens(717.2104, 717.4816, 735.3566, 552.7982, 
                   -0.1389272, -0.001239606, 0.0009125824, -0.00004071615);
  
  cv::Mat im1 = imread("images/1.png", cv::IMREAD_COLOR);
  std::vector<cv::Point2f> points;
  std::vector<cv::Vec3b> colors_original;
  std::vector<cv::Point3f> image3d;
  std::cout << "Image size: " << im1.cols << "x" << im1.rows << std::endl;
  int resolution = 1;
  for (int x = 0; x < im1.cols; x+=resolution) {
    for (int y = 0; y < im1.rows; y+=resolution) {
      double xd = (x - lens.cx()) / lens.fx();
      double yd = (y - lens.cy()) / lens.fy();
      points.push_back(cv::Point2f(x, y));
      colors_original.push_back(im1.at<cv::Vec3b>(y, x));
      image3d.push_back(cv::Point3f(-xd, -yd, -lens.f()));
    }
  }

  // Project back the image points onto the semi-sphere
  std::vector<std::vector<double>> coords3d_reconstr;
  //double f = 1.0; //lens.f();
  for (auto coord : points) {
    double x = coord.x;
    double y = coord.y;
    std::vector<double> coord3d = lens.Compute3D(x, y, false, 1.0);
    coords3d_reconstr.push_back(coord3d);
  }

  // Save points to csv
  std::ofstream myfile;
  myfile.open("/home/icirauqui/w0rkspace/CV/fisheye_lens_cc/points.csv");
  for (auto coord : coords3d_reconstr) {
    myfile << coord[0] << "," << coord[1] << "," << coord[2] << std::endl;
  }
  myfile.close();




  // - - - - Visualization - - - - - - - - - - - - - - - - - -
 
  float scale = 1;
  float offset = 0.75;

  // Projected image
  std::vector<cv::Point3f> points_image;
  std::vector<cv::Vec3b> colors_image;
  for (auto coord: image3d) {
    double x = scale * coord.x;
    double y = scale * coord.y;
    double z = - offset;
    points_image.push_back(cv::Point3f(x, y, z));
    colors_image.push_back(cv::Vec3b(50, 50, 150));
  }  

  // Reconstructed lens
  std::vector<cv::Point3f> points_lens_reconstr;
  for (auto coord: coords3d_reconstr) {
    double x = scale * sin(coord[0]) * cos(coord[1]);
    double y = scale * sin(coord[0]) * sin(coord[1]);
    double z = offset + scale * cos(coord[0]);
    if (z - offset < 0) {
      x = 0.0;
      y = 0.0;
      z = 0.0;
    }
    points_lens_reconstr.push_back(cv::Point3f(x, y, z));
  }

  Visualizer vis;
  vis.AddCloud(points_image, colors_original);
  vis.AddCloud(points_lens_reconstr, colors_original);
  vis.Render();
}






int main() {

  //FullLens();

  //ImgMethod();

  return 0;
}
