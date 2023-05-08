#include <iostream>
#include <fstream>
#include <math.h>

#include <cstdlib>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/viz.hpp>

#include "aux.h"
#include "ang_matcher.h"
#include "../third_party/nlohmann/json.hpp"
#include "camera.h"

#include "fe_lens/fe_lens.hpp"



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



int menu() {
  int option = 0;
  std::cout << std::endl;
  std::cout << " Select display option: " << std::endl;
  std::cout << "  1. Epiline - Sampson" << std::endl;
  std::cout << "  2. Epiline - Angle2D" << std::endl;
  std::cout << "  3. Epiline - Angle3D" << std::endl;
  std::cout << "  4. Sampson - Angle2D" << std::endl;
  std::cout << "  5. Epiline - Angle3D" << std::endl;
  std::cout << "  6. Angle2D - Angle3D" << std::endl;
  std::cout << "  7. All" << std::endl;
  std::cout << "  9. Exit" << std::endl << std::endl;

  std::cout << " Option: ";
  std::cin >> option;
  std::cout << std::endl;
  return option;
}


using namespace am;




int main() {
  std::cout << " 1. Loading data" << std::endl; 

  FisheyeLens lens(717.2104, 717.4816, 735.3566, 552.7982, 
                   -0.1389272, -0.001239606, 0.0009125824, -0.00004071615);

  std::cout << " 1.1. Camera parameters from cams.json" << std::endl;
  Camera cam = Camera("images/cams.json");

  std::cout << " 1.2. Images" << std::endl;

  int num_pairs = 0;
  std::string img_path = "";
  std::vector<std::vector<std::string>> image_pairs;

  //std::ifstream json_file("images/imgs.json");
  std::ifstream json_file("images/in/pairs.json");
  nlohmann::json json_data = nlohmann::json::parse(json_file);
  if (json_data.empty()) {
    std::cout << "Unable to load parameters from images.json" << std::endl;
  } else {
    nlohmann::json im_control = json_data["control"];
    num_pairs = im_control["num_pairs"];
    img_path = im_control["path"];

    nlohmann::json im_pairs = json_data["image_pairs"];

    if (!im_pairs.empty()) {
      for (auto i : im_pairs) {
        std::vector<std::string> pair;
        pair.push_back(i[0]);
        pair.push_back(i[1]);
        image_pairs.push_back(pair);
      }
    }

    std::cout << "      " << im_control["num_pairs"] << " image pairs available" << std::endl;
    for (unsigned int i = 0; i < image_pairs.size(); i++) {
      std::cout << "       · " << image_pairs[i][0] << " - " << image_pairs[i][1] << std::endl;
    }
  }

  std::vector<std::vector<cv::Mat>> images;
  for (unsigned int i = 0; i < image_pairs.size(); i++) {
    std::vector<cv::Mat> pair;
    pair.push_back(cv::imread(img_path + image_pairs[i][0], cv::IMREAD_COLOR));
    pair.push_back(cv::imread(img_path + image_pairs[i][1], cv::IMREAD_COLOR));
    images.push_back(pair);
  }

  cv::Mat im1 = imread("images/s1_001.png", cv::IMREAD_COLOR);
  cv::Mat im2 = imread("images/s1_002.png", cv::IMREAD_COLOR);

  float f = cam.FocalLength();
  cv::Point3f c1 = cam.CameraCenter();
  cv::Point3f c2 = cam.CameraCenter();

  


  std::cout << " 2. Detecting features" << std::endl;

  // Detect features (parameters from COLMAP)
  int max_features = 8192; //8192;
  int num_octaves = 4;
  int octave_resolution = 3;
  float peak_threshold = 0.02 / octave_resolution;  // 0.04
  float edge_threshold = 10;
  float sigma = 1.6;
  
  cv::Ptr<cv::SIFT> f2d = cv::SIFT::create(max_features, num_octaves, peak_threshold, edge_threshold, sigma);
  std::vector<cv::KeyPoint> kps1, kps2;
  cv::Mat desc1, desc2;
  f2d->detect(im1, kps1, cv::noArray());
  f2d->detect(im2, kps2, cv::noArray());

  // Remove keypoints from contour and compute descriptors
  // Compute contour, erode its shape and remove the kps outside it
  std::vector<std::vector<cv::Point>> contours1 = GetContours(im1, 20, 3, false);
  std::vector<std::vector<cv::Point>> contours2 = GetContours(im2, 20, 3, false);
  kps1 = KeypointsInContour(contours1[0], kps1);
  kps2 = KeypointsInContour(contours2[0], kps2);
  f2d->compute(im1, kps1, desc1);
  f2d->compute(im2, kps2, desc2);






  std::cout << " 3. Matching features" << std::endl;

  std::vector<cv::DMatch> matches_knn = MatchKnn(desc1, desc2, 0.8f);
  std::vector<cv::DMatch> matches_knn_07 = MatchKnn(desc1, desc2, 0.7f);
  std::vector<cv::DMatch> matches_flann = MatchFLANN(desc1, desc2, 0.8f);
  std::vector<cv::DMatch> matches_flann_07 = MatchFLANN(desc1, desc2, 0.7f);
  std::vector<cv::DMatch> matches_bf = MatchBF(desc1, desc2, true);

  // Select the matches to use (COLMAP defaults)
  std::vector<cv::DMatch> matches = MatchFLANN(desc1, desc2, 0.5f);
  SaveMatches(matches, "/home/icirauqui/workspace_phd/fisheye_matcher/images/matches.txt");

  std::cout << " 3.1. Matches: " << matches.size() << std::endl;



  std::cout << " 4. Compute F and epilines" << std::endl;

  // Compute F and epilines
  std::vector<cv::Point2f> points1, points2;
  for (unsigned int i = 0; i < matches.size(); i++) {
    points1.push_back(kps1[matches[i].queryIdx].pt);
    points2.push_back(kps2[matches[i].trainIdx].pt);
  }
  cv::Mat F12 = cv::findFundamentalMat(points1, points2);

  std::cout << " 4.1 Decompose E" << std::endl;
  cv::Mat Kp = cam.K();
  Kp.convertTo(Kp, CV_64F);
  cv::Mat E = EfromF(F12, Kp);

  cv::Mat R1, R2, t;
  cv::decomposeEssentialMat(E, R1, R2, t);

  cv::Point3f c1g(0.0, 0.0, 0.0);
  cv::Point3f c2g = c1g + cv::Point3f(t.at<double>(0, 0), t.at<double>(0, 1), t.at<double>(0, 2));




  std::cout << " 5. Compute matches by distance and angle" << std::endl;
  
  bool cross_check = true;
  bool draw_inline = false;
  bool draw_global = false;
  
  //float th_epiline = 4.0;
  float th_sampson = 4.0;
  //float th_angle2d = DegToRad(1.0);
  float th_angle3d = DegToRad(2.0);
  double th_sift = 100.0;

  // Match by distance threshold
  AngMatcher am(kps1, kps2, 
                desc1, desc2, F12, im1, im2, 
                2*cam.Cx(), 2*cam.Cy(), f, 
                c1, c2, 
                c1g, c2g, 
                R1, R2, t, 
                cam.K(), cam.D(), 
                &lens);

  am.Match("sampson", th_sampson, th_sift, cross_check, draw_inline, draw_global);
  am.Match("angle3d", th_angle3d, th_sift, cross_check, draw_inline, draw_global);

  //am.ViewMatches("epiline", "epiline desc matches", 0.5);





  std::cout << " 6. Compare matches" << std::endl;
  int report_level = 3;
  am.CompareMatches("sampson", "angle3d", report_level);


  std::cout << " 7. Compare matches for specific query keypoint" << std::endl;
  am.ViewCandidatesCompare("sampson", "angle3d", 605);
  am.ViewCandidatesCompareLines("sampson", "angle3d", 605);
  
  
  

  // ----------------------------------------------------------------------------------------------
  // ----------------------------------------------------------------------------------------------
  // ----------------------------------------------------------------------------------------------
  // ----------------------------------------------------------------------------------------------
  // ----------------------------------------------------------------------------------------------

  /*
  
  // Create the image for compare
  cv::Mat img1 = im1.clone();
  cv::Mat img2 = im2.clone();  
  float opacity = 0.5;

  std::vector<std::vector<cv::Point2f>> points_candidate_th_pi_intersect;


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
    //vis.AddCloud(points_images[i], imgs[i].colors_);
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




  */


  cv::waitKey(0);
  return 0;
}
