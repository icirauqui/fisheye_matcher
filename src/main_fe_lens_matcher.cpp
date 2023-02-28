#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp> // include the viz module

#include <fstream>


#include "fe_lens/fe_lens.hpp"
#include "matcher/matcher.hpp"


#include "src/ang_matcher/ang_matcher.h"




void SaveMatches(std::vector<cv::DMatch> matches, std::string path) {
  std::ofstream file;
  file.open(path);
  for (int i = 0; i < matches.size(); i++) {
    file << matches[i].queryIdx << " " << matches[i].trainIdx << " " << matches[i].distance << std::endl;
  }
  file.close();
}

std::vector<cv::DMatch> LoadMatches(std::string path) {
  std::vector<cv::DMatch> matches;
  std::ifstream file(path);
  std::string line;
  while (std::getline(file, line)) {
    std::istringstream iss(line);
    int idx1, idx2;
    float dist;
    if (!(iss >> idx1 >> idx2 >> dist)) { break; } // error
    matches.push_back(cv::DMatch(idx1, idx2, dist));
  }
  return matches;
}



void ImgMatching() {
  FisheyeLens lens(717.2104, 717.4816, 735.3566, 552.7982, 
                   -0.1389272, -0.001239606, 0.0009125824, -0.00004071615);

  // Load images  
  std::cout << " 1. Loading images" << std::endl;
  std::vector<Image> imgs;
  imgs.push_back(Image(imread("images/s1_001.png", cv::IMREAD_COLOR), &lens));
  imgs.push_back(Image(imread("images/s1_002.png", cv::IMREAD_COLOR), &lens));



  std::cout << " 2. Detecting features" << std::endl;

  // Detect features (parameters from COLMAP)
  int max_features = 8192; //8192;
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

  //std::vector<cv::DMatch> matches = am::MatchFLANN(imgs[0].desc_, imgs[1].desc_, 0.7f);
  std::vector<cv::DMatch> matches = LoadMatches("/home/icirauqui/workspace_phd/fisheye_matcher/images/matches.txt");
  
  std::cout << " 3.1. Matches: " << matches.size() << std::endl;



  std::cout << " 4. Compute F and epilines" << std::endl;

  // Compute F and epilines
  std::vector<cv::Point2f> points1, points2;
  for (unsigned int i = 0; i < matches.size(); i++) {
    points1.push_back(imgs[0].kps_[matches[i].queryIdx].pt);
    points2.push_back(imgs[1].kps_[matches[i].trainIdx].pt);
  }
  cv::Mat F12 = cv::findFundamentalMat(points1, points2);
  std::cout << "      F12: " << F12 << std::endl;

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

  std::cout << " 4.2. R1: " << R1 << std::endl;
  std::cout << " 4.3. R2: " << R2 << std::endl;
  std::cout << " 4.4. t: " << t << std::endl;

  double tx = t.at<double>(0, 0);
  double ty = t.at<double>(0, 1);
  double tz = t.at<double>(0, 2);

  
  
  std::cout << " 5. Compute matches by distance and angle" << std::endl;
    
  bool cross_check = true;
  bool draw_inline = false;
  bool draw_global = false;
  
  //float th_epiline = 4.0;
  float th_sampson = 4.0;
  //float th_angle2d = DegToRad(1.0);
  float th_angle3d = DegToRad(2.0);
  double th_sift = 100.0;

  cv::Point3f c1g(0.0, 0.0, 0.0);
  cv::Point3f c2g = c1g + cv::Point3f(tx, ty, tz);

  std::cout << " 6. Compare matches" << std::endl;

  // Different points: 32, 395, 409, 430, 473, 644
  // Points with high FE impact 605, 804, 2893, 6478
  // Points with low FE impact 3948, 6506

  int point_analysis = 605;

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
    std::cout << "    Match angle found" << std::endl;
    cv::circle(img2, cv::Point2f(match_angle.x, match_angle.y), 12, cv::Vec3b(255,255,255), 3);
    cv::circle(img2, cv::Point2f(match_angle.x, match_angle.y), 9, cv::Vec3b(0,255,0), -1);
  } else {
    std::cout << "    Match angle not found" << std::endl;
  }

  if (match_sampson.x > 0 && match_sampson.y > 0) {
    std::cout << "    Match sampson found" << std::endl;
    cv::circle(img2, cv::Point2f(match_sampson.x, match_sampson.y), 12, cv::Vec3b(255,255,255), 3);
    cv::circle(img2, cv::Point2f(match_sampson.x, match_sampson.y), 9, cv::Vec3b(0,0,255), -1);
  } else {
    std::cout << "    Match sampson not found" << std::endl;
  }

  if (match_angle.x == match_sampson.x && match_angle.y == match_sampson.y) {
    std::cout << "    Match angle and sampson are the same" << std::endl;
  } else {
    std::cout << "    Match angle and sampson are different" << std::endl;
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
  cv::imwrite("/home/icirauqui/workspace_phd/fisheye_matcher/images/img12.png", img12);

  vis.Render();
}



int main() {

  //FullLens();

  //ImgMethod();

  ImgMatching();

  return 0;
}
