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
#include "third_party/nlohmann/json.hpp"
#include "camera.h"




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
  int max_features = 1000; //8192;
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

  //float max_ratio = 0.8f;           //COLMAP
  //float max_distance = 0.7f;        //COLMAP
  //int max_num_matches = 32768;      //COLMAP
  //float max_error = 4.0f;           //COLMAP
  //float confidence = 0.999f;        //COLMAP
  //int max_num_trials = 10000;       //COLMAP
  //float min_inliner_ratio = 0.25f;  //COLMAP
  //int min_num_inliers = 15;         //COLMAP
  
  bool cross_check = true;
  bool draw_inline = false;
  bool draw_global = false;

  std::vector<cv::DMatch> matches_knn = MatchKnn(desc1, desc2, 0.8f);
  std::vector<cv::DMatch> matches_knn_07 = MatchKnn(desc1, desc2, 0.7f);
  std::vector<cv::DMatch> matches_flann = MatchFLANN(desc1, desc2, 0.8f);
  std::vector<cv::DMatch> matches_flann_07 = MatchFLANN(desc1, desc2, 0.7f);
  std::vector<cv::DMatch> matches_bf = MatchBF(desc1, desc2, true);

  std::cout << " 3.1. Knn   | 0.7 | 0.8 :\t" << matches_knn_07.size()   << "\t|\t" << matches_knn.size() << std::endl;
  std::cout << " 3.2. Flann | 0.7 | 0.8 :\t" << bold_on << matches_flann_07.size() << bold_off << "\t|\t" << matches_flann.size() << std::endl;
  std::cout << " 3.3. BF                :\t" << matches_bf.size()       << "\t|\t" << std::endl;

  // Select the matches to use (COLMAP defaults)
  std::vector<cv::DMatch> matches = matches_flann_07;




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
  
  float th_epiline = 4.0;
  float th_sampson = 4.0;
  float th_angle2d = DegToRad(1.0);
  float th_angle3d = DegToRad(5.0);
  double th_sift = 100.0;

  // Match by distance threshold
  AngMatcher am(kps1, kps2, desc1, desc2, F12, im1, im2, 2*cam.Cx(), 2*cam.Cy(), f, c1, c2, c1g, c2g, R1, R2, t, cam.K(), cam.D());

  //am.Match("epiline", th_epiline, th_sift, cross_check, draw_inline, draw_global);
  am.Match("sampson", th_sampson, th_sift, cross_check, draw_inline, draw_global);
  //am.Match("angle2d", th_angle2d, th_sift, cross_check, draw_inline, draw_global);
  //am.Match("angle3d", th_angle3d, th_sift, cross_check, draw_inline, draw_global);
  //am.Match("angle3d", DegToRad(1.0), th_sift, cross_check, draw_inline, draw_global);
  am.Match("angle3d", DegToRad(4.0), th_sift, cross_check, draw_inline, draw_global);

  //am.ViewMatches("epiline", "epiline desc matches", 0.5);





  std::cout << " 6. Compare matches" << std::endl;
  int report_level = 3;
  //am.CompareMatches("epiline", "sampson", report_level);
  //am.CompareMatches("epiline", "angle2d", report_level);
  //am.CompareMatches("epiline", "angle3d", report_level);
  //am.CompareMatches("sampson", "angle2d", report_level);
  am.CompareMatches("sampson", "angle3d", report_level);
  //am.CompareMatches("angle2d", "angle3d", report_level);


  std::cout << " 7. Compare matches for specific query keypoint" << std::endl;
  am.ViewCandidatesCompare("sampson", "angle3d",  32);
  am.ViewCandidatesCompare("sampson", "angle3d", 395);
  am.ViewCandidatesCompare("sampson", "angle3d", 409);
  am.ViewCandidatesCompare("sampson", "angle3d", 430);
  am.ViewCandidatesCompare("sampson", "angle3d", 473);
  am.ViewCandidatesCompare("sampson", "angle3d", 642);
  am.ViewCandidatesCompare("sampson", "angle3d", 644);




  //am.ViewCandidates("epiline",  217, "epiline");
  //am.ViewCandidates("epiline",  804, "epiline");
  //am.ViewCandidates("epiline", 3158, "epiline");
  //am.ViewCandidates("epiline", 6358, "epiline");
  //am.ViewCandidates("angle2d",  217, "angle2d");
  //am.ViewCandidates("angle2d",  804, "angle2d");
  //am.ViewCandidates("angle2d", 3158, "angle2d");
  //am.ViewCandidates("angle2d", 6358, "angle2d");

  //am.ViewCandidates("sampson", 4982, "sampson");
  //am.ViewCandidates("sampson", 4982, "sampson");
  //am.ViewCandidates("angle2d", 5081, "angle2d");
  //am.ViewCandidates("angle2d", 5081, "angle2d");





/*
  int option = 0;
  while (option != 9) {

    option = menu();
    system("clear");

    cv::destroyAllWindows();

    if (option == 1) 
      am.CompareMatches("epiline", "sampson", report_level);
    else if (option == 2) 
      am.CompareMatches("epiline", "angle2d", report_level);
    else if (option == 3) 
      am.CompareMatches("epiline", "angle3d", report_level);
    else if (option == 4) 
      am.CompareMatches("sampson", "angle2d", report_level);
    else if (option == 5) 
      am.CompareMatches("sampson", "angle3d", report_level);
    else if (option == 6) 
      am.CompareMatches("angle2d", "angle3d", report_level);
    else if (option == 7) {
      am.CompareMatches("epiline", "sampson", report_level);
      am.CompareMatches("epiline", "angle2d", report_level);
      am.CompareMatches("epiline", "angle3d", report_level);
      am.CompareMatches("sampson", "angle2d", report_level);
      am.CompareMatches("sampson", "angle3d", report_level);
      am.CompareMatches("angle2d", "angle3d", report_level);
    }
    else if (option == 9) 
      break;

    cv::waitKey(0);
  }
*/







  // For a point matched by Sampson and not by angle
  // Draw the epipolar line in the second image, and the point







/*

  std::cout << " 8. Draw ressults" << std::endl;


  cv::Mat imout_matches_knn, imout_matches_sampson, imout_matches_angle;

  cv::drawMatches(im1, kps1, im2, kps2, matches_knn, imout_matches_knn);
  cv::drawMatches(im1, kps1, im2, kps2, matches_sampson, imout_matches_sampson);
  cv::drawMatches(im1, kps1, im2, kps2, matches_angle, imout_matches_angle);

  std::vector<cv::Mat> ims = {imout_matches_knn, imout_matches_sampson, imout_matches_angle};
  ResizeAndDisplay("Matches", ims, 0.4);

  //ResizeAndDisplay("Matches KNN", imout_matches_knn, 0.5);
  //ResizeAndDisplay("Matches Sampson", imout_matches_sampson, 0.5);
  //ResizeAndDisplay("Matches Angle", imout_matches_angle, 0.5);

*/


  cv::waitKey(0);
  return 0;
}
