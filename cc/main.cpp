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
  cv::Point3f c2 = cam.CameraCenter();




  std::cout << " 3. Detecting features" << std::endl;

  // Detect features (parameters from COLMAP)
  int max_features = 8192;
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
  



  /*

  std::cout << " Compare epiline drawing" << std::endl;
  // Get Points from KeyPoints
  std::vector<cv::Point2f> kpoints1, kpoints2;
  for (size_t i = 0; i < kps1.size(); i++)
    kpoints1.push_back(kps1[i].pt);
  for (size_t i = 0; i < kps2.size(); i++)
    kpoints2.push_back(kps2[i].pt);

  // Compute epilines with given F
  std::vector<cv::Vec3f> gmlines1, gmlines2;
  cv::computeCorrespondEpilines(kpoints1, 1, F12, gmlines1);
  cv::computeCorrespondEpilines(kpoints2, 2, F12, gmlines2);

  std::cout << "Man " << gmlines1[0][0] << " " << gmlines1[0][1] << " " << gmlines1[0][2] << std::endl;

  std::vector<cv::Vec3f> gmline;
  gmline.push_back(gmlines1[0]);

  float lx = 2*im1.cols; //2*cam.Cx();
  float ly = 2*im1.rows; //cam.Cy();

  // Manually draw line
  cv::Point3f pt0(0, -gmlines1[0][2] / gmlines1[0][1], 0.);
  cv::Point3f pt1(lx, -(gmlines1[0][2] + gmlines1[0][0] * lx) / gmlines1[0][1], 0.);
  cv::Point3f pt2(im1.cols, -(gmlines1[0][2] + gmlines1[0][0] * im1.cols) / gmlines1[0][1], 0.);
  cv::Vec3f line = EquationLine(cv::Point2f(pt0.x, pt0.y), cv::Point2f(pt1.x, pt1.y));

  //cv::Point(        0, -epilines1[i][2] / epilines1[i][1]),
  //cv::Point(img1.cols, -(epilines1[i][2] + epilines1[i][0] * img1.cols) / epilines1[i][1]),


  std::vector<cv::Point2f> skp1, skp2;
  skp1.push_back(kpoints1[0]);
  skp2.push_back(kpoints2[0]);
  cv::Mat im_epi = CompareEpipolarLines("epip1",F12,im1,im2,skp1,skp2);

  std::cout << " abc" << std::endl;

  cv::line(im_epi, cv::Point2f(im1.cols + pt0.x, pt0.y), cv::Point2f(im1.cols + pt1.x, pt1.y), cv::Scalar(0, 0, 255), 2);
  cv::line(im_epi, cv::Point2f(im1.cols + pt0.x, pt0.y), cv::Point2f(im1.cols + pt2.x, pt2.y), cv::Scalar(0, 255, 0), 2);

  std::cout << "def" << std::endl;

  ResizeAndDisplay("abc", im_epi, 0.5);
  cv::waitKey(0);

  return 0;

  */









  std::cout << " 6. Compute matches by distance and angle" << std::endl;
  //float lx = im1.cols;
  //float ly = im1.rows;

  float th_alpha = DegToRad(1.0);
  float th_sampson = 10.;
  double th_sift = 100.0;

  // Match by distance threshold
  std::vector<std::vector<double>> matches_sampson_all = MatchSampson(kps1, kps2, desc1, desc2, F12, 2*cam.Cx(), 2*cam.Cy(), c2, th_sampson, true, false);
  std::vector<cv::DMatch> matches_sampson = NNCandidates(matches_sampson_all, th_sift);

  
  // Match by angle threshold
  std::vector<std::vector<double>> matches_angle_all = MatchAngle(kps1, kps2, desc1, desc2, F12, 2*cam.Cx(), 2*cam.Cy(), c2, th_alpha, true, false);
  std::vector<cv::DMatch> matches_angle = NNCandidates(matches_angle_all, th_sift);

  // Match by distance threshold
  std::cout << " 6.1. Sampson all/nn: " << matches_sampson_all.size() << " / " << matches_sampson.size() << std::endl;
  std::cout << " 6.2. Angle all/nn:   " << matches_angle_all.size()   << " / " << matches_angle.size() << std::endl;



  return 0;








  std::cout << " 7. Draw ressults" << std::endl;


  cv::Mat imout_matches_knn, imout_matches_sampson, imout_matches_angle;

  cv::drawMatches(im1, kps1, im2, kps2, matches_knn, imout_matches_knn);
  cv::drawMatches(im1, kps1, im2, kps2, matches_sampson, imout_matches_sampson);
  cv::drawMatches(im1, kps1, im2, kps2, matches_angle, imout_matches_angle);

  std::vector<cv::Mat> ims = {imout_matches_knn, imout_matches_sampson, imout_matches_angle};
  ResizeAndDisplay("Matches", ims, 0.4);

  //ResizeAndDisplay("Matches KNN", imout_matches_knn, 0.5);
  //ResizeAndDisplay("Matches Sampson", imout_matches_sampson, 0.5);
  //ResizeAndDisplay("Matches Angle", imout_matches_angle, 0.5);

  // HistogramDMatch("Matches KNN    ",matches_knn,th_sift,10);
  // HistogramDMatch("Matches Sampson",matches_sampson,th_sift,10);
  // HistogramDMatch("Matches Angle  ",matches_angle,th_sift,10);




  cv::waitKey(0);
  return 0;
}
