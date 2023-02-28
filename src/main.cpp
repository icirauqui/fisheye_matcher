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



std::vector<cv::DMatch> FEMatcher() {
  FisheyeLens lens(717.2104, 717.4816, 735.3566, 552.7982, 
                 -0.1389272, -0.001239606, 0.0009125824, -0.00004071615);

  std::cout << " 1. Loading images" << std::endl;
  cv::Mat im1 = imread("images/s1_001.png", cv::IMREAD_COLOR);
  cv::Mat im2 = imread("images/s1_002.png", cv::IMREAD_COLOR);

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
  std::vector<std::vector<cv::Point>> contours1 = am::GetContours(im1, 20, 3, false);
  std::vector<std::vector<cv::Point>> contours2 = am::GetContours(im2, 20, 3, false);
  kps1 = am::KeypointsInContour(contours1[0], kps1);
  kps2 = am::KeypointsInContour(contours2[0], kps2);
  f2d->compute(im1, kps1, desc1);
  f2d->compute(im2, kps2, desc2);

  std::cout << " 3. Matching features" << std::endl;
  std::vector<cv::DMatch> matches = LoadMatches("matches.txt");
  std::cout << " 3.1. Matches: " << matches.size() << std::endl;

  // Draw matches
  cv::Mat im_matches;
  cv::drawMatches(im1, kps1, im2, kps2, matches, im_matches);

  // Resize image
  cv::resize(im_matches, im_matches, cv::Size(), 0.5, 0.5);

  cv::imshow("Matches FE", im_matches);


  return matches;
}


std::vector<cv::DMatch> AngMatcher() {
  FisheyeLens lens(717.2104, 717.4816, 735.3566, 552.7982, 
                 -0.1389272, -0.001239606, 0.0009125824, -0.00004071615);

  std::cout << " 1. Loading images" << std::endl;
  cv::Mat im1 = imread("images/s1_001.png", cv::IMREAD_COLOR);
  cv::Mat im2 = imread("images/s1_002.png", cv::IMREAD_COLOR);

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
  std::vector<std::vector<cv::Point>> contours1 = am::GetContours(im1, 20, 3, false);
  std::vector<std::vector<cv::Point>> contours2 = am::GetContours(im2, 20, 3, false);
  kps1 = am::KeypointsInContour(contours1[0], kps1);
  kps2 = am::KeypointsInContour(contours2[0], kps2);
  f2d->compute(im1, kps1, desc1);
  f2d->compute(im2, kps2, desc2);

  std::cout << " 3. Matching features" << std::endl;
  std::vector<cv::DMatch> matches = am::MatchFLANN(desc1, desc2, 0.7f);
  std::cout << " 3.1. Matches: " << matches.size() << std::endl;

  SaveMatches(matches, "matches.txt");


  // Draw matches
  cv::Mat im_matches;
  cv::drawMatches(im1, kps1, im2, kps2, matches, im_matches);

  // Resize image
  cv::resize(im_matches, im_matches, cv::Size(), 0.5, 0.5);

  cv::imshow("Matches Ang", im_matches);


  return matches;
}



int main() {
  std::vector<cv::DMatch> matches_ang_1 = AngMatcher();
  std::cout << std::endl << std::endl;

  std::vector<cv::DMatch> matches_fe_1 = FEMatcher();
  std::cout << std::endl << std::endl;

  cv::waitKey(0);

  return 0;
}
