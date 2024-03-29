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

  //FisheyeLens lens(717.2104, 717.4816, 735.3566, 552.7982, -0.1389272, -0.001239606, 0.0009125824, -0.00004071615);
  FisheyeLens lens(717.691,  718.021,  734.728,  552.072,  -0.139659,  -0.000313999, 0.0015055,    -0.000131671);

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


  // Undistort images
  //for (unsigned int r=0; r<im1.rows; r++) {
  //  for (unsigned int c=0; c<im1.cols; c++) {
  //    cv::Point2f p = lens.Undistort(cv::Point2f(c, r));
  //    im1.at<cv::Vec3b>(r, c) = images[0][0].at<cv::Vec3b>(p.y, p.x);
  //  }
  //}
  


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
  //float max_ratio = 0.8f;           //COLMAP
  //float max_distance = 0.7f;        //COLMAP
  //int max_num_matches = 32768;      //COLMAP
  //float max_error = 4.0f;           //COLMAP
  //float confidence = 0.999f;        //COLMAP
  //int max_num_trials = 10000;       //COLMAP
  //float min_inliner_ratio = 0.25f;  //COLMAP
  //int min_num_inliers = 15;         //COLMAP

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



  std::cout << " 6. Compare matches" << std::endl;
  int report_level = 3;
  am.CompareMatches("sampson", "angle3d", report_level);
  am.ViewCandidatesCompare("sampson", "angle3d", 605);
  am.ViewCandidatesCompareLines("sampson", "angle3d", 605);

  // Create the image for compare
  cv::Mat img1 = im1.clone();
  cv::Mat img2 = im2.clone();  
  float opacity = 0.5;





























  cv::waitKey(0);
  return 0;
}
