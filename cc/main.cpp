#include <iostream>
#include <fstream>
#include <math.h>

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
#include "feature_matcher.h"



using namespace am;


void onTrackbar(int, void*) {
  // Nothing
}


int tr_sampson = 1;
int tr_angle2d = 1;
int tr_epiline = 1;
int tr_angle3d = 1;

void on_tr_sampson(int, void*) {

}
void on_tr_angle2d(int, void*) {

}
void on_tr_epiline(int, void*) {

}
void on_tr_angle3d(int, void*) {

}




int main() {
  bool bDraw = false;


  std::cout << " 1. Loading parameters from cams.json" << std::endl;

  Camera cam = Camera("../images/cams.json");
  cv::Mat K = cam.K();
  cv::Vec4f D = cam.D();


  std::cout << " 2. Loading images" << std::endl;

  cv::Mat im1 = imread("../images/1.png", cv::IMREAD_COLOR);
  cv::Mat im2 = imread("../images/2.png", cv::IMREAD_COLOR);

  float f = cam.FocalLength();
  cv::Point3f c1 = cam.CameraCenter();
  cv::Point3f c2 = cam.CameraCenter();

  






  std::cout << " 3. Detecting features" << std::endl;

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





  std::cout << " 4. Matching features" << std::endl;

  float max_ratio = 0.8f;
  float max_distance = 0.7f;
  bool cross_check = true;
  int max_num_matches = 32768;
  float max_error = 4.0f;
  float confidence = 0.999f;
  int max_num_trials = 10000;
  float min_inliner_ratio = 0.25f;
  int min_num_inliers = 15;

  std::vector<cv::DMatch> matches_knn = MatchKnn(desc1, desc2, 0.8f);
  std::vector<cv::DMatch> matches_knn_07 = MatchKnn(desc1, desc2, 0.7f);
  std::vector<cv::DMatch> matches_flann = MatchFLANN(desc1, desc2, 0.8f);
  std::vector<cv::DMatch> matches_flann_07 = MatchFLANN(desc1, desc2, 0.7f);
  std::vector<cv::DMatch> matches_bf = MatchBF(desc1, desc2, true);

  std::cout << " 4.1. Knn   | 0.7 | 0.8 :\t" << matches_knn_07.size()   << "\t|\t" << matches_knn.size() << std::endl;
  std::cout << " 4.2. Flann | 0.7 | 0.8 :\t" << bold_on << matches_flann_07.size() << bold_off << "\t|\t" << matches_flann.size() << std::endl;
  std::cout << " 4.3. BF                :\t" << matches_bf.size()       << "\t|\t" << std::endl;

  // Select the matches to use
  std::vector<cv::DMatch> matches = matches_flann_07;




  std::cout << " 5. Compute F and epilines" << std::endl;

  // Compute F and epilines
  std::vector<cv::Point2f> points1, points2;
  for (unsigned int i = 0; i < matches.size(); i++) {
    points1.push_back(kps1[matches[i].queryIdx].pt);
    points2.push_back(kps2[matches[i].trainIdx].pt);
  }
  cv::Mat F12 = cv::findFundamentalMat(points1, points2);

  //DrawEpipolarLines("epip1",F12,im1,im2,points1,points2);
  //cv::waitKey(0);


  std::cout << " 5.1 Decompose E" << std::endl;
  cv::Mat Kp = cam.K();
  Kp.convertTo(Kp, CV_64F);
  cv::Mat E = EfromF(F12, Kp);

  cv::Mat R1, R2, t;
  cv::decomposeEssentialMat(E, R1, R2, t);

  cv::Mat c1p = -R1.t() * t;

  cv::Point3f c1g(0.0, 0.0, 0.0);
  cv::Point3f c2g = c1g + cv::Point3f(t.at<double>(0, 0), t.at<double>(0, 1), t.at<double>(0, 2));









  std::cout << " 6. Compute matches by distance and angle" << std::endl;
  std::cout << " 6.1. Sampson original" << std::endl;
  std::cout << " 6.2. Angle original" << std::endl;
  std::cout << " 6.3. Epiline new" << std::endl;
  std::cout << " 6.4. Angle new" << std::endl;
  
  float th_sampson = 4.0;
  float th_angle2d = DegToRad(1.0);
  float th_epiline = 4.0;
  float th_angle3d = DegToRad(1.0);



  cv::namedWindow("Controls", cv::WINDOW_NORMAL);
  cv::createTrackbar("Sampson [  1  - 10 ]", "Controls", &tr_sampson, 10, on_tr_sampson);
  cv::createTrackbar("Angle2d [ 0.1 -  1 ]", "Controls", &tr_angle2d, 10, on_tr_angle2d);
  cv::createTrackbar("Epiline [  1  - 10 ]", "Controls", &tr_epiline, 10, on_tr_epiline);
  cv::createTrackbar("Angle3d [ 0.1 -  1 ]", "Controls", &tr_angle3d, 10, on_tr_angle3d);


  double th_sift = 100.0;

  // Match by distance threshold
  AngMatcher am(kps1, kps2, desc1, desc2, F12, im1, im2, 2*cam.Cx(), 2*cam.Cy(), f, c1, c2, c1g, c2g, R1, R2, t, K);

  std::vector<std::vector<double>> matches_sampson_all = am.Match("sampson", th_sampson, true, false, false);
  std::vector<std::vector<double>> matches_angle2d_all = am.Match("angle2d", th_angle2d, true, false, false);
  std::vector<std::vector<double>> matches_epiline_all = am.Match("epiline", th_epiline, true, false, false);
  std::vector<std::vector<double>> matches_angle3d_all = am.Match("angle3d", th_angle3d, true, false, false);

  am.View(matches_sampson_all, CountMaxIdx(matches_sampson_all), "matches_sampson_all");
  am.View(matches_angle2d_all, CountMaxIdx(matches_angle2d_all), "matches_angle2d_all");
  am.View(matches_epiline_all, CountMaxIdx(matches_epiline_all), "matches_epiline_all");
  am.View(matches_angle3d_all, CountMaxIdx(matches_angle3d_all), "matches_angle3d_all");

  std::vector<cv::DMatch> matches_sampson = am.NNCandidates(matches_sampson_all, th_sift);
  std::vector<cv::DMatch> matches_angle2d = am.NNCandidates(matches_angle2d_all, th_sift);
  std::vector<cv::DMatch> matches_epiline = am.NNCandidates(matches_epiline_all, th_sift);
  std::vector<cv::DMatch> matches_angle3d = am.NNCandidates(matches_angle3d_all, th_sift);

  std::vector<cv::DMatch> matches_sampson_desc = am.MatchDescriptors(matches_sampson_all, desc1, desc2, th_sift);
  std::vector<cv::DMatch> matches_angle2d_desc = am.MatchDescriptors(matches_angle2d_all, desc1, desc2, th_sift);
  std::vector<cv::DMatch> matches_epiline_desc = am.MatchDescriptors(matches_epiline_all, desc1, desc2, th_sift);
  std::vector<cv::DMatch> matches_angle3d_desc = am.MatchDescriptors(matches_angle3d_all, desc1, desc2, th_sift);

  std::cout << " 6.1. Sampson all/nn/desc: " << am::CountPositive(matches_sampson_all) << " / " << matches_sampson.size() << " / " << matches_sampson_desc.size() << std::endl;
  std::cout << " 6.2. Angle2d all/nn/desc: " << am::CountPositive(matches_angle2d_all) << " / " << matches_angle2d.size() << " / " << matches_angle2d_desc.size() << std::endl;
  std::cout << " 6.3. Epiline all/nn/desc: " << am::CountPositive(matches_epiline_all) << " / " << matches_epiline.size() << " / " << matches_epiline_desc.size() << std::endl;
  std::cout << " 6.4. Angle3d all/nn/desc: " << am::CountPositive(matches_angle3d_all) << " / " << matches_angle3d.size() << " / " << matches_angle3d_desc.size() << std::endl;

  cv::Mat im_matches_sampson_desc; cv::drawMatches(im1, kps1, im2, kps2, matches_sampson_desc, im_matches_sampson_desc);
  cv::Mat im_matches_angle2d_desc; cv::drawMatches(im1, kps1, im2, kps2, matches_angle2d_desc, im_matches_angle2d_desc);
  cv::Mat im_matches_epiline_desc; cv::drawMatches(im1, kps1, im2, kps2, matches_epiline_desc, im_matches_epiline_desc);
  cv::Mat im_matches_angle3d_desc; cv::drawMatches(im1, kps1, im2, kps2, matches_angle3d_desc, im_matches_angle3d_desc);

  ResizeAndDisplay("matches_sampson_desc", im_matches_sampson_desc, 0.5);
  ResizeAndDisplay("matches_angle2d_desc", im_matches_angle2d_desc, 0.5);
  ResizeAndDisplay("matches_epiline_desc", im_matches_epiline_desc, 0.5);
  ResizeAndDisplay("matches_angle3d_desc", im_matches_angle3d_desc, 0.5);

  cv::waitKey(0);
  




/*
  std::cout << " 7. Compare matches" << std::endl;

  FeatureMatcher fm(im1, im2);

  cv::Mat imout_matches_segregation = fm.CompareMatches(
    im1, im2, 
    kps1, kps2,
    matches_sampson, matches_angle2d, 
    1);

  ResizeAndDisplay("Matches Segregation", imout_matches_segregation, 0.5);


  cv::Mat imout_matches_candidates = fm.DrawCandidates(
    im1, im2, 
    kps1, kps2,
    F12,
    matches_sampson, matches_angle2d, 
    1);



  ResizeAndDisplay("Epipolar compare", imout_matches_candidates, 0.5);
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
