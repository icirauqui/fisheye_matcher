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







void ImgMatching() {
  FisheyeLens lens(717.2104, 717.4816, 735.3566, 552.7982, 
                   -0.1389272, -0.001239606, 0.0009125824, -0.00004071615);

  // Load images  
  std::cout << " 1. Loading images" << std::endl;
  std::vector<Image> imgs;
  imgs.push_back(Image(imread("images/1.png", cv::IMREAD_COLOR), &lens));
  imgs.push_back(Image(imread("images/2.png", cv::IMREAD_COLOR), &lens));



  std::cout << " 2. Detecting features" << std::endl;

  // Detect features (parameters from COLMAP)
  int max_features = 1000; //8192;
  int num_octaves = 4;
  int octave_resolution = 3;
  float peak_threshold = 0.02 / octave_resolution;  // 0.04
  float edge_threshold = 10;
  float sigma = 1.6;

  cv::Ptr<cv::SIFT> f2d = cv::SIFT::create(max_features, num_octaves, peak_threshold, edge_threshold, sigma);
  for (auto &img : imgs) {
    f2d->detect(img.image_, img.kps_, cv::noArray());
    img.contours_ = am::GetContours(img.image_, 20, 3, false);
    img.kps_ = am::KeypointsInContour(img.contours_[0], img.kps_);
    f2d->compute(img.image_, img.kps_, img.desc_);
    //std::cout << "      Keypoints: " << img.kps_.size() 
    //          << "      Descriptors: " << img.desc_.rows << std::endl;
  }
  



  std::cout << " 3. Matching features" << std::endl;

  //float max_ratio = 0.8f;           //COLMAP
  //float max_distance = 0.7f;        //COLMAP
  //int max_num_matches = 32768;      //COLMAP
  //float max_error = 4.0f;           //COLMAP
  //float confidence = 0.999f;        //COLMAP
  //int max_num_trials = 10000;       //COLMAP
  //float min_inliner_ratio = 0.25f;  //COLMAP
  //int min_num_inliers = 15;         //COLMAP

  std::vector<cv::DMatch> matches = am::MatchFLANN(imgs[0].desc_, imgs[1].desc_, 0.7f);
  std::cout << " 3.1. Matches: " << matches.size() << std::endl;



  std::cout << " 4. Compute F and epilines" << std::endl;

  // Compute F and epilines
  std::vector<cv::Point2f> points1, points2;
  for (unsigned int i = 0; i < matches.size(); i++) {
    points1.push_back(imgs[0].kps_[matches[i].queryIdx].pt);
    points2.push_back(imgs[1].kps_[matches[i].trainIdx].pt);
  }
  cv::Mat F12 = cv::findFundamentalMat(points1, points2);

  std::cout << " 4.1 Decompose E" << std::endl;
  cv::Mat Kp = lens.K();
  Kp.convertTo(Kp, CV_64F);
  cv::Mat E = am::EfromF(F12, Kp);

  cv::Mat R1, R2, t;
  cv::decomposeEssentialMat(E, R1, R2, t);
  imgs[0].R_ = R1;
  imgs[0].t_ = cv::Mat::zeros(3, 1, CV_64F);
  imgs[1].R_ = R2;
  imgs[1].t_ = t;

  //std::cout << " 4.2. R1" << std::endl;
  //std::cout << R1 << std::endl;
  //std::cout << " 4.3. R2" << std::endl;
  //std::cout << R2 << std::endl;
  //std::cout << " 4.4. t" << std::endl;
  //std::cout << t << std::endl;


  double tx = t.at<double>(0, 0);
  double ty = t.at<double>(0, 1);
  double tz = t.at<double>(0, 2);

  
  







  std::cout << " 5. Compute matches by distance and angle" << std::endl;
    
  bool cross_check = true;
  bool draw_inline = false;
  bool draw_global = false;
  
  //float th_epiline = 4.0;
  float th_sampson = 8.0;
  //float th_angle2d = DegToRad(1.0);
  float th_angle3d = DegToRad(2.0);
  double th_sift = 2000.0;

  cv::Point3f c1g(0.0, 0.0, 0.0);
  cv::Point3f c2g = c1g + cv::Point3f(tx, ty, tz);

  // Match by distance threshold
  //am::AngMatcher angmatcher(&imgs[0], &imgs[1], 
  //                          &lens, 
  //                          F12, 
  //                          R1, R2, t,
  //                          c1g, c2g);








  std::cout << " 6. Compare matches" << std::endl;

  // Different points: 32, 395, 409, 430, 473, 642, 644
  int point_analysis = 473;


  cv::Point2f pt1 = imgs[0].kps_[point_analysis].pt;
  std::vector<double> pt1_cil_cam = lens.Compute3D(pt1.x, pt1.y, false);
  cv::Point3d pt1_cam = lens.CilToCart(pt1_cil_cam[0], pt1_cil_cam[1]);
  cv::Point3d pt1_w = imgs[0].PointGlobal(pt1_cam);
  cv::Vec4f pi1 = am::EquationPlane(pt1_w, c1g, c2g);






  // - - - - Visualization - - - - - - - - - - - - - - - - - -
 
  float scale = 1;
  cv::Vec3d offset(0, 0, 0);
  float th_intersect = 0.005;

  std::vector<std::vector<cv::Point3f>> points_images;
  for (unsigned int i=0; i<imgs.size(); i++) {
    // Projected image
    std::vector<cv::Point3f> points_image;
    for (auto coord_img: imgs[i].image3d_) {
      cv::Point3f coord = imgs[i].PointGlobal(cv::Point3f(coord_img.x, coord_img.y, 0.0));
      cv::Point3f offset_i = cv::Point3f(i*offset(0), i*offset(1), i*offset(2));
      points_image.push_back(offset_i + scale*coord);
    }  
    points_images.push_back(points_image);
  }



  // Draw each lens, its intersection with the epipolar plane, and the search region
  std::vector<std::vector<cv::Point3f>> points_lens;
  std::vector<std::vector<cv::Point3f>> points_lens_pi_intersect;
  std::vector<std::vector<cv::Point3f>> points_candidate_th;
  std::vector<std::vector<cv::Point2f>> points_candidate_th_pi_intersect;

  for (unsigned int i=0; i<imgs.size(); i++) {
    std::vector<cv::Point3f> points_lens_reconstr;
    std::vector<cv::Point3f> points_lens_pi_intersect_i;
    std::vector<cv::Point3f> points_candidate_th_i;
    std::vector<cv::Point2f> points_candidate_th_pi_intersect_i;

    int j = 0;
    for (auto coord_img_c: imgs[i].lens_->Lens3dReconstr()) {
      cv::Point3f coord_img = lens.CilToCart(coord_img_c[0], coord_img_c[1]);
      
      if (coord_img.z < 0) {
        coord_img = cv::Point3f(0.0, 0.0, 0.0);
      }

      cv::Point3f coord = imgs[i].PointGlobal(coord_img);
      cv::Point3f offset_i = cv::Point3f(i*offset(0), i*offset(1), i*offset(2));
      points_lens_reconstr.push_back(offset_i + scale*coord);

      // Check if point is on epipolar plane
      double error_pi = pi1(0)*coord.x + pi1(1)*coord.y + pi1(2)*coord.z + pi1(3);
      if (std::abs(error_pi) < th_intersect) {
        points_lens_pi_intersect_i.push_back(coord);
      }

      // Check if point is under angle threshold
      cv::Point3f coord_th = coord - cv::Point3f(tx, ty, tz);
      double angle = am::AngleLinePlane(pi1, coord_th);
      if (angle < th_angle3d) {
        points_candidate_th_i.push_back(coord);
        cv::Point2f pt_over_image = imgs[i].points_[j];
        points_candidate_th_pi_intersect_i.push_back(pt_over_image);
      }
      j++;
    }

    points_lens.push_back(points_lens_reconstr);
    points_lens_pi_intersect.push_back(points_lens_pi_intersect_i);
    points_candidate_th.push_back(points_candidate_th_i);
    points_candidate_th_pi_intersect.push_back(points_candidate_th_pi_intersect_i);
  }








  Matcher matcher_angle;
  matcher_angle.MatchAngle(&lens, &imgs[0], &imgs[1], th_angle3d);
  std::vector<cv::Point3f> points_candidates_angle = matcher_angle.candidates_crosscheck_[point_analysis];
  std::vector<cv::DMatch> candidates_angle_nn = matcher_angle.NNMatches(th_sift);
  std::vector<cv::DMatch> candidates_angle_desc = matcher_angle.DescMatches(&imgs[0], &imgs[1], th_sift);

  std::vector<cv::Point3f> points_candidates_angle_nn = 
    matcher_angle.Match3D(&imgs[1], &lens, candidates_angle_nn, point_analysis);
  
  std::vector<cv::Point3f> points_candidates_angle_desc =
    matcher_angle.Match3D(&imgs[1], &lens, candidates_angle_desc, point_analysis);

  cv::Point2f match_angle = matcher_angle.Match2D(&imgs[1], &lens, candidates_angle_desc, point_analysis);


  // Compute and draw epipolar candidates
  Matcher matcher_sampson;
  matcher_sampson.MatchSampson(&lens, &imgs[0], &imgs[1], F12, th_sampson);
  std::vector<cv::Point3f> candidates_sampson = matcher_sampson.candidates_crosscheck_[point_analysis];
  std::vector<cv::DMatch> candidates_sampson_nn = matcher_sampson.NNMatches(th_sift);
  std::vector<cv::DMatch> candidates_sampson_desc = matcher_sampson.DescMatches(&imgs[0], &imgs[1], th_sift);

  std::vector<cv::Point3f> points_candidates_sampson_desc =
    matcher_angle.Match3D(&imgs[1], &lens, candidates_sampson_desc, point_analysis);

  cv::Point2f match_sampson = matcher_sampson.Match2D(&imgs[1], &lens, candidates_sampson_desc, point_analysis);






  // In image 2, draw the angle threshold and the epipolar line threshold
  // Draw points_candidate_th[1] over img2 with opacity 0.5
  Visualizer vis;  
  float opacity = 0.5;
  cv::Mat img1 = imgs[0].image_.clone();
  cv::Mat img2 = imgs[1].image_.clone();

  // Draw intersection of epipolar plane with lens in image 2
  for (auto pt: points_candidate_th_pi_intersect[1]) {
    img2.at<cv::Vec3b>(pt.y, pt.x) = img2.at<cv::Vec3b>(pt.y, pt.x) * (1-opacity) + cv::Vec3b(0,255,0) * opacity;
  }



  // Draw epipolar line and search region
  std::vector<cv::Point2f> points_epipolar_th = matcher_sampson.SampsonRegion(&lens, &imgs[0], &imgs[1], F12, point_analysis, th_sampson);
  for (auto pt: points_epipolar_th) {
    img2.at<cv::Vec3b>(pt.y, pt.x) = img2.at<cv::Vec3b>(pt.y, pt.x) * (1-opacity) + cv::Vec3b(0,0,255) * opacity;
  }




  // Draw Angle candidates in image 2
  std::vector<double> points_candidates_val_angle = matcher_angle.candidates_val_crosscheck_[point_analysis];
  for (unsigned int i=0; i<points_candidates_val_angle.size(); i++) {
    if (points_candidates_val_angle[i] > 0.) {
      cv::Point2f pt_over_image = imgs[1].kps_[i].pt;
      cv::circle(img2, cv::Point2f(pt_over_image.x, pt_over_image.y), 5, vis.yellow, -1);
    }
  }



  // Draw Sampson canddates in image 2
  for (auto pt: candidates_sampson) {
    cv::circle(img2, cv::Point2f(pt.x, pt.y), 5, vis.blue, -1);
  }

  // Draw query point in image 1
  cv::circle(img1, cv::Point2f(pt1.x, pt1.y), 10, vis.blue, 2);




  // Draw match if exists

  if (match_angle.x > 0 && match_angle.y > 0) {
    std::cout << "Match angle found" << std::endl;
    cv::circle(img2, cv::Point2f(match_angle.x, match_angle.y), 12, cv::Vec3b(255,255,255), 3);
    cv::circle(img2, cv::Point2f(match_angle.x, match_angle.y), 9, cv::Vec3b(0,255,0), -1);
  } else {
    std::cout << "Match angle not found" << std::endl;
  }

  if (match_sampson.x > 0 && match_sampson.y > 0) {
    std::cout << "Match sampson found" << std::endl;
    cv::circle(img2, cv::Point2f(match_sampson.x, match_sampson.y), 12, cv::Vec3b(255,255,255), 3);
    cv::circle(img2, cv::Point2f(match_sampson.x, match_sampson.y), 9, cv::Vec3b(0,0,255), -1);
  } else {
    std::cout << "Match sampson not found" << std::endl;
  }

  if (match_angle.x == match_sampson.x && match_angle.y == match_sampson.y) {
    std::cout << "Match angle and sampson are the same" << std::endl;
  } else {
    std::cout << "Match angle and sampson are different" << std::endl;
  }










  for (unsigned int i=0; i<points_images.size(); i++) {
    vis.AddCloud(points_images[i], imgs[i].colors_);
    vis.AddCloud(points_lens[i], imgs[i].colors_);

    vis.AddCloud(points_candidate_th[i], cv::Vec3d(0,0,255), 1.0, 0.2);
    vis.AddCloud(points_lens_pi_intersect[i], cv::Vec3d(0,0,0), 1.0, 1.0);
  }

  std::cout << "Candidates angle all/nn/desc: " 
            << points_candidates_angle.size() << " / " 
            << candidates_angle_nn.size() << "-" << points_candidates_angle_nn.size() << " / "
            << candidates_angle_desc.size() << "-" << points_candidates_angle_desc.size() << std::endl;

  if (points_candidates_angle.size() > 0)
    vis.AddCloud(points_candidates_angle, vis.yellow, 3.0);
  if (points_candidates_angle_nn.size() > 0)
    vis.AddCloud(points_candidates_angle_nn, vis.green, 6.0);
  if (points_candidates_angle_desc.size() > 0)
    vis.AddCloud(points_candidates_angle_desc, vis.pink, 6.0);


  // Point in image 1
  std::vector<cv::Point3f> point_im_1 = {pt1_w};
  vis.AddCloud(point_im_1, vis.green, 6.0);

  vis.AddPlane(pt1_w, c1g, c2g, 5);









  // Before visualizing, save the image pair
  cv::Mat img12;
  cv::hconcat(img1, img2, img12);
  cv::resize(img12, img12, cv::Size(), 0.5, 0.5);  
  cv::imwrite("/home/icirauqui/w0rkspace/CV/fisheye_lens_cc/images/img12.png", img12);
  //cv::imshow("img2", img12);
  //cv::waitKey(0);







  vis.Render();




/*


  // - - - - Visualization - - - - - - - - - - - - - - - - - -
 
  float scale = 1;
  cv::Vec3d offset(0, 0, 0);

  std::vector<std::vector<cv::Point3f>> points_images;
  for (unsigned int i=0; i<images.size(); i++) {
    // Projected image
    std::vector<cv::Point3f> points_image;
    for (auto coord: image3d) {
      double x = (i*tx) + (i*offset(0)) + scale * coord.x;
      double y = (i*ty) + (i*offset(1)) + scale * coord.y;
      double z = (i*tz) + (i*offset(2)) + 0.0;
      points_image.push_back(cv::Point3f(x, y, z));
    }  
    points_images.push_back(points_image);
  }



  // Reconstructed lens
  std::vector<cv::Point3f> points_lens_reconstr;
  for (auto coord: images_3d[0]) {
    double x = scale * sin(coord[0]) * cos(coord[1]);
    double y = scale * sin(coord[0]) * sin(coord[1]);
    double z = scale * cos(coord[0]);
    if (z < 0) {
      x = 0.0;
      y = 0.0;
      z = 0.0;
    }
    points_lens_reconstr.push_back(cv::Point3f(x, y, z));
  }

  Visualizer vis;
  vis.AddCloud(points_images[0], colors_original);
  vis.AddCloud(points_images[1], colors_original);
  vis.AddCloud(points_lens_reconstr, colors_original);
  vis.Render();

  */
}



int main() {

  //FullLens();

  //ImgMethod();

  ImgMatching();

  return 0;
}
