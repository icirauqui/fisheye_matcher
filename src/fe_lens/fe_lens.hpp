#ifndef FE_LENS_HPP
#define FE_LENS_HPP

#include <iostream>
#include <vector>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp> // include the viz module

#include <boost/math/tools/polynomial.hpp>
#include <boost/math/tools/roots.hpp>


class NewtonRaphson {

// Instructions for Newton-Raphson solver
//   double f(double x) { return x * x - 2; }
//   double f_prime(double x) { return 2 * x; }
//   NewtonRaphson solver(1, 1e-6, 100);
//   double x = solver.solve(f, f_prime);

public:
  NewtonRaphson(double tol, unsigned int max_iter);

  double solve(double x0, std::function<double (double)> (f), std::function<double (double)> (f_prime));

private:
  double tol_;
  unsigned int max_iter_;
};




class FisheyeLens {
public:
  FisheyeLens(double fx, double fy, double cx, double cy,
              double k1, double k2, double k3, double k4);

  double RTheta(double theta);

  double Rd(double theta);

  // Inverse of RTheta with Newton-Raphson
  double RThetaInv(double r_theta, double x0 = 0.1);

  cv::Point2f Compute2D(double theta, double phi, bool world_coord = false);

  std::vector<double> Compute3D(double x, double y, bool world_coord = false, double x0 = 0.1);

  std::vector<double> Get3D(int x, int y, bool world_coord = false);

  double ComputeError(std::vector<std::vector<double>> v1, std::vector<std::vector<double>> v2);

  double f();

  cv::Point3d c();

  cv::Mat K();

  cv::Vec4d D();

  bool Lens3dReconstrEmpty();

  void Lens3dReconstr(std::vector<cv::Point2f> points,
                      int w = 0, int h = 0);

  cv::Point3d CilToCart(double theta, double phi, double r = 1.0);

  std::vector<double> CartToCil(double x, double y, double z);

  inline std::vector<std::vector<double>> Lens3dReconstr() { return lens3d_reconstr_; }

  inline double fx() { return fx_; }
  inline double fy() { return fy_; }
  inline double cx() { return cx_; }
  inline double cy() { return cy_; }
  inline double k1() { return k1_; }
  inline double k2() { return k2_; }
  inline double k3() { return k3_; }
  inline double k4() { return k4_; }
  inline double k5() { return k5_; }

private:
  double fx_ = 0.0;
  double fy_ = 0.0;
  double cx_ = 0.0;
  double cy_ = 0.0;
  double k1_ = 0.0;
  double k2_ = 0.0;
  double k3_ = 0.0;
  double k4_ = 0.0;
  double k5_ = 0.0;

  std::vector<std::vector<double>> lens3d_reconstr_;
  std::vector<std::vector<std::vector<double>>> lens3d_reconstr_grid_;
  std::vector<double> thetas_;
};



class Image {

public:
  Image(cv::Mat image, FisheyeLens* lens);

  cv::Point3f PointGlobal(cv::Point3f pt);

  cv::Point3f PointGlobal(cv::Point3f pt, cv::Mat R, cv::Mat t);

  cv::Point3f PointGlobalRotation(cv::Point3f pt);

  cv::Point3f PointGlobalTranslation(cv::Point3f pt);

  inline cv::Mat GetImage() { return image_; }

  cv::Point3f cg();

  std::vector<cv::Point2f> points_;
  std::vector<cv::Vec3b> colors_;
  std::vector<cv::Point3f> image3d_;
  std::vector<std::vector<double>> coords3d_reconstr_;
  
  cv::Mat image_;
  FisheyeLens* lens_;

  std::vector<cv::KeyPoint> kps_;
  std::vector<std::vector<cv::Point>> contours_;
  cv::Mat desc_;

  cv::Mat R_;
  cv::Mat t_;
};



class Visualizer {

public: 
  Visualizer(std::string window_name = "3D", double scale = 1.0);

  void AddCloud(std::vector<cv::Point3f> cloud, std::vector<cv::Vec3b> color, 
                double scale = 1.0, double opacity = 1.0);

  void AddCloud(std::vector<cv::Point3f> cloud, cv::Vec3b color = cv::Vec3b(100,100,100),
                double scale = 1.0, double opacity = 1.0);

  void AddWidget(cv::viz::Widget widget);

  void AddPlane(std::vector<double> kp, cv::Point3f pt1, cv::Point3f pt2, int size = 1);

  void AddPlane(cv::Point3f pt0, cv::Point3f pt1, cv::Point3f pt2, int size = 1);

  void Render();

  // Colors 
  cv::Vec3d yellow = cv::Vec3d(0,255,255);
  cv::Vec3d red = cv::Vec3d(255,0,0);
  cv::Vec3d green = cv::Vec3d(0,255,0);
  cv::Vec3d blue = cv::Vec3d(255,0,0);
  cv::Vec3d black = cv::Vec3d(0,0,0);
  cv::Vec3d pink = cv::Vec3d(255,0,255);

private:

  std::vector<std::vector<cv::Point3f>> point_clouds_;
  std::vector<std::vector<cv::Vec3b>> colors_;
  std::vector<double> scales_;
  std::vector<double> opacities_;

  std::vector<cv::viz::Widget> widgets_;

  cv::viz::Viz3d window_;
  std::string window_name_ = "3D";
  double scale_ = 1.0;

};

#endif

