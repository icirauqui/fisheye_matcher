#include "fe_lens.hpp"



double vis_scale = 1.0;

NewtonRaphson::NewtonRaphson(double tol, unsigned int max_iter) : tol_(tol), max_iter_(max_iter) {}


double NewtonRaphson::solve(double x0, std::function<double (double)> (f), std::function<double (double)> (f_prime)) {
  double x = x0;

  //std::cout << "Newton-Raphson: " << x << std::endl;
    
  for (unsigned int i=0; i<max_iter_; i++) {
    double fx = f(x);
    double dfx = f_prime(x);
    double x1 = x - fx / dfx;
    //std::cout << "NR " << i << ":\t" << x << " " << fx << " " << dfx << " " << x1 << std::endl;
    if (std::abs(x1 - x) < tol_) {
      return x1;
    }
    x = x1;
  }
  return std::numeric_limits<double>::quiet_NaN(); // return NaN if the method doesn't converge
}



FisheyeLens::FisheyeLens(double fx, double fy, double cx, double cy,
              double k1, double k2, double k3, double k4): 
              fx_(fx), fy_(fy), cx_(cx), cy_(cy), k1_(k1), k2_(k2), k3_(k3), k4_(k4) {}


double FisheyeLens::RTheta(double theta) {
  return k1_*theta + k2_*pow(theta,3) + k3_*pow(theta,5) + k4_*pow(theta,7);
}


double FisheyeLens::Rd(double theta) {
  return theta * ( 1 + k1_*pow(theta,2) + k2_*pow(theta,4) + k3_*pow(theta,6) + k4_*pow(theta,8) );
}


double FisheyeLens::RThetaInv(double r_theta, double x0) {
  //std::cout << "RThetaInv: " << r_theta << std::endl;
//  auto func = [this, r_theta](double th) {
//    return k1_*pow(th,1) + k2_*pow(th,3) + k3_*pow(th,5) + k4_*pow(th,7) - r_theta;
//  };
//
//  auto func_prime = [this](double th) {
//    return k1_*1*pow(th,0) - k2_*3*pow(th,2) - k3_*5*pow(th,4) - k4_*7*pow(th,6);
//  };

  double r_d = r_theta;

  auto func = [this, r_d](double th) {
    return  th * ( 1.0 + k1_*pow(th,2) + k2_*pow(th,4) + k3_*pow(th,6) + k4_*pow(th,8)) - r_d;
  };

  auto func_prime = [this](double th) {
    return  1.0 + 3*k1_*pow(th,2) + 5*k2_*pow(th,4) + 7*k3_*pow(th,6) + 9*k4_*pow(th,8);
  };

  NewtonRaphson solver(1e-6, 1000);
  double theta_solver = solver.solve(x0, func, func_prime);

  return theta_solver;
}



cv::Point2f FisheyeLens::Compute2D(double theta, double phi, bool world_coord) {
  double r_d = Rd(theta);
  double x_d = r_d * cos(phi);
  double y_d = r_d * sin(phi);

  if (world_coord) {
    return cv::Point2f(x_d, y_d);
  } else {
    double x = fx_ * x_d + cx_;
    double y = fy_ * y_d + cy_;
    return cv::Point2f(x, y);
  }
}


std::vector<double> FisheyeLens::Compute3D(double x, double y, bool world_coord, double x0) {  
  double phi = 0.0;
  double theta = 0.0;

  double x_d = x;
  double y_d = y;

  // If we're in camera coordinate system, convert to world coordinate system
  if (!world_coord) {
    x_d = (x - cx_) / fx_;
    y_d = (y - cy_) / fy_;
  }

  // If the point is at the center of the image, return 0,0
  if (x_d == 0.0 && y_d == 0.0) {
    return std::vector<double>({theta, phi});
  }

  // If the point is on the y-axis, set phi to pi/2 or -pi/2
  if (x_d == 0.0) {
    phi = (y_d >= 0.0) ? (M_PI / 2.0) : (-M_PI / 2.0);
  } else {
    phi = atan(y_d / x_d);
  }

  // If the point is on the negative x-axis, sum 180 degrees to phi
  if (x_d < 0.0) {
    phi += M_PI;
  }

  // Compute the distance from the center of the image to the distorted point
  double r_d = sqrt(pow(x_d, 2) + pow(y_d, 2));

  theta = RThetaInv(r_d, x0);
            
  return std::vector<double>({theta, phi});
}


std::vector<double> FisheyeLens::Get3D(int x, int y, bool world_coord) {
  return lens3d_reconstr_grid_[y][x];
}


double FisheyeLens::ComputeError(std::vector<std::vector<double>> v1, std::vector<std::vector<double>> v2) {
  double error_theta = 0.0;
  double error_phi = 0.0;

  for (unsigned int i = 0; i < v1.size(); i++) {
    double error_theta_i = pow(v1[i][0] - v2[i][0], 2);
    double error_phi_i = pow(v1[i][1] - v2[i][1], 2);
    error_theta += error_theta_i;
    error_phi += error_phi_i;
  }
  error_theta = sqrt(error_theta / v1.size());
  error_phi = sqrt(error_phi / v1.size());
  std::cout << "Error theta: " << error_theta << std::endl;
  std::cout << "Error phi: " << error_phi << std::endl;

  return sqrt(pow(error_theta, 2) + pow(error_phi, 2));
}



double FisheyeLens::f() {
  double lx = 2 * cx_;
  double ly = 2 * cy_;

  double f = (lx / (lx + ly)) * fx_ + (ly / (lx + ly)) * fy_;

  return f;
}


cv::Point3d FisheyeLens::c() {
  double f = this->f();
  return cv::Point3d(cx_, cy_, f);
}


cv::Mat FisheyeLens::K() {
  cv::Mat K = cv::Mat::zeros(3, 3, CV_64F);
  K.at<double>(0, 0) = fx_;
  K.at<double>(1, 1) = fy_;
  K.at<double>(0, 2) = cx_;
  K.at<double>(1, 2) = cy_;
  K.at<double>(2, 2) = 1.0;

  // Convert K to CV_64F
  K.convertTo(K, CV_64F);

  return K;
}


cv::Vec4d FisheyeLens::D() {
  return cv::Vec4f(k1_, k2_, k3_, k4_);
}

bool FisheyeLens::Lens3dReconstrEmpty() {
  if (lens3d_reconstr_.size() == 0) {
    return true;
  }
  return false;
}


void FisheyeLens::Lens3dReconstr(std::vector<cv::Point2f> points,
                                 int w, int h) {
  // Project back the image points onto the semi-sphere
  if (w != 0 && h != 0) {
    lens3d_reconstr_grid_ = std::vector<std::vector<std::vector<double>>>(w, std::vector<std::vector<double>>(h, std::vector<double>(2, 0.0)));  
  }

  for (auto coord : points) {
    double x = coord.x;
    double y = coord.y;
    std::vector<double> coord3d = Compute3D(x, y, false, 1.0);
    lens3d_reconstr_.push_back(coord3d);

    if (w != 0 && h != 0) {
      lens3d_reconstr_grid_[x][y] = coord3d;
    }
  }

  if (w != 0 && h != 0) {
    double w_max = std::max(cx_, w - cx_);
    double h_max = std::max(cy_, h - cy_);
    double r_max = sqrt(pow(w_max, 2) + pow(h_max, 2));

    for (unsigned int i=0; i<r_max; i++) {
      std::vector<double> coord3d = Compute3D(i/fx_, 0, true, 1.0);
      thetas_.push_back(coord3d[0]);
    }
  }



}


cv::Point3d FisheyeLens::CilToCart(double theta, double phi, double r) {
  double x = r * sin(theta) * cos(phi);
  double y = r * sin(theta) * sin(phi);
  double z = r * cos(theta);

  return cv::Point3d(x, y, z);
}


std::vector<double> FisheyeLens::CartToCil(double x, double y, double z) {
  double r = sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2));
  double theta = acos(z / r);
  double phi = atan2(y, x);

  return std::vector<double>({theta, phi});


}


Image::Image(cv::Mat image, FisheyeLens* lens):
  image_(image), lens_(lens) {

  // Image points
  int resolution = 1;
  for (int x = 0; x < image_.cols; x+=resolution) {
    for (int y = 0; y < image_.rows; y+=resolution) {
      double xd = (x - lens->cx()) / lens->fx();
      double yd = (y - lens->cy()) / lens->fy();
      points_.push_back(cv::Point2f(x, y));
      colors_.push_back(image_.at<cv::Vec3b>(y, x));
      image3d_.push_back(cv::Point3f(-xd, -yd, -lens->f()));
    }
  }

  if (lens->Lens3dReconstrEmpty()) {
    lens->Lens3dReconstr(points_, image_.cols, image_.rows);
  }
}


cv::Point3f Image::PointGlobal(cv::Point3f pt) {
  cv::Mat p = (cv::Mat_<double>(3,1) << pt.x, pt.y, pt.z);
  cv::Mat p2 = R_ * p + t_;
  cv::Point3f coord(p2.at<double>(0, 0), p2.at<double>(1, 0), p2.at<double>(2, 0));
  return coord;
}


cv::Point3f Image::PointGlobal(cv::Point3f pt, cv::Mat R, cv::Mat t) {
  cv::Mat p = (cv::Mat_<double>(3,1) << pt.x, pt.y, pt.z);
  cv::Mat p2 = R * p + t;
  cv::Point3f coord(p2.at<double>(0, 0), p2.at<double>(1, 0), p2.at<double>(2, 0));
  return coord;
}


cv::Point3f Image::PointGlobalRotation(cv::Point3f pt) {
  cv::Mat p = (cv::Mat_<double>(3,1) << pt.x, pt.y, pt.z);
  cv::Mat p2 = R_ * p;
  cv::Point3f coord(p2.at<double>(0, 0), p2.at<double>(1, 0), p2.at<double>(2, 0));
  return coord;
}


cv::Point3f Image::PointGlobalTranslation(cv::Point3f pt) {
  cv::Mat p = (cv::Mat_<double>(3,1) << pt.x, pt.y, pt.z);
  cv::Mat p2 = p + t_;
  cv::Point3f coord(p2.at<double>(0, 0), p2.at<double>(1, 0), p2.at<double>(2, 0));
  return coord;
}


cv::Point3f Image::cg() {
  double tx = t_.at<double>(0, 0);
  double ty = t_.at<double>(0, 1);
  double tz = t_.at<double>(0, 2);
  return cv::Point3f(tx, ty, tz);
}


Visualizer::Visualizer(std::string window_name, double scale):
  window_name_(window_name), scale_(scale) {
  //cv::viz::Viz3d window_(window_name_);
  //cv::namedWindow(window_name_, cv::WINDOW_AUTOSIZE);
}


void Visualizer::AddCloud(std::vector<cv::Point3f> cloud, std::vector<cv::Vec3b> color, 
                          double scale, double opacity) {
  point_clouds_.push_back(cloud);
  colors_.push_back(color);
  scales_.push_back(scale);
  opacities_.push_back(opacity);
}


void Visualizer::AddCloud(std::vector<cv::Point3f> cloud, cv::Vec3b color, 
                          double scale, double opacity) {
  std::vector<cv::Vec3b> colors;
  for (unsigned int i = 0; i < cloud.size(); i++) {
    colors.push_back(color);
  }
  AddCloud(cloud, colors, scale, opacity);
}


void Visualizer::AddWidget(cv::viz::Widget widget) {
  widgets_.push_back(widget);
}


void Visualizer::AddPlane(std::vector<double> kp, cv::Point3f pt1, cv::Point3f pt2, int size) {
  // Draw a plane in 3D that goes through c1g, c2g and kp3d
  double kp3d_theta = kp[0];
  double kp3d_phi = kp[1];
  double kp3d_x = sin(kp3d_theta) * cos(kp3d_phi);
  double kp3d_y = sin(kp3d_theta) * sin(kp3d_phi);
  double kp3d_z = cos(kp3d_theta);
  cv::Point3f kp3d(kp3d_x, kp3d_y, kp3d_z);
  cv::Point3f n = (pt2 - pt1).cross(kp3d - pt1);
  n = n / cv::norm(n);

  cv::viz::WPlane plane_widget(cv::Point3d(0, 0, 0), 
                               cv::Vec3d(n.x, n.y, n.z), 
                               cv::Vec3d(0, 0, 1),
                               cv::Size2d(size, size));
  plane_widget.setRenderingProperty(cv::viz::LINE_WIDTH, 2.0);
  plane_widget.setRenderingProperty(cv::viz::OPACITY, 0.5);
  
  AddWidget(plane_widget);
}


void Visualizer::AddPlane(cv::Point3f pt0, cv::Point3f pt1, cv::Point3f pt2, int size) {
  // Draw a plane in 3D that goes through c1g, c2g and kp3d
  cv::Point3f n = (pt2 - pt1).cross(pt0 - pt1);
  n = n / cv::norm(n);

  cv::viz::WPlane plane_widget(cv::Point3d(0, 0, 0), 
                               cv::Vec3d(n.x, n.y, n.z), 
                               cv::Vec3d(0, 0, 1),
                               cv::Size2d(size, size));

  plane_widget.setRenderingProperty(cv::viz::LINE_WIDTH, 2.0);
  plane_widget.setRenderingProperty(cv::viz::OPACITY, 0.5);
  
  AddWidget(plane_widget);
}


void Visualizer::Render() {
  window_ = cv::viz::Viz3d(window_name_);

  for (unsigned int i=0; i<point_clouds_.size(); i++) {
    std::string name = "cloud" + std::to_string(i);
    cv::viz::WCloud cloud(point_clouds_[i], colors_[i]);
    if (scales_[i] != 1.0) {
      cloud.setRenderingProperty(cv::viz::POINT_SIZE, scales_[i]);
    }
    if (opacities_[i] != 1.0) {
      cloud.setRenderingProperty(cv::viz::OPACITY, opacities_[i]);
    }
    window_.showWidget(name, cloud);
  }

  for (unsigned int i=0; i<widgets_.size(); i++) {
    std::string name = "widget" + std::to_string(i);
    window_.showWidget(name, widgets_[i]);
  }

  // Add a coordinate system
  cv::viz::WCoordinateSystem world_coor(vis_scale);
  window_.showWidget("World", world_coor);

  window_.spin();
}



