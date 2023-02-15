#ifndef ANG_MATCHER_H
#define ANG_MATCHER_H


#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/viz.hpp>

#include "aux.h"



namespace am {


cv::Mat EfromF(const cv::Mat &F, const cv::Mat &K);

void RtfromEsvd(const cv::Mat &E, cv::Mat &R, cv::Mat &t);

void RtfromE(const cv::Mat &E, cv::Mat &K, cv::Mat &R, cv::Mat &t);

std::vector<cv::DMatch> MatchKnn(const cv::Mat &descriptors1, const cv::Mat &descriptors2, float ratio_thresh);

std::vector<cv::DMatch> MatchFLANN(const cv::Mat &descriptors1, const cv::Mat &descriptors2, float ratio_thresh);

std::vector<cv::DMatch> MatchBF(const cv::Mat &descriptors1, const cv::Mat &descriptors2, bool crossCheck = false);



cv::Mat CompareEpipolarLines(const std::string &title, const cv::Mat F,
                              const cv::Mat &img1, const cv::Mat &img2,
                              const std::vector<cv::Point2f> points1, const std::vector<cv::Point2f> points2,
                              const float inlierDistance = -1);


void DrawEpipolarLines(const std::string &title, const cv::Mat F,
                              const cv::Mat &img1, const cv::Mat &img2,
                              const std::vector<cv::Point2f> points1, const std::vector<cv::Point2f> points2,
                              const float inlierDistance = -1);


void DrawEpipolarLines(const std::string &title, const cv::Mat F,
                              const cv::Mat &img1, const cv::Mat &img2,
                              const std::vector<cv::Point2f> points1,
                              const std::vector<cv::Vec3f> epilines1,
                              const std::vector<cv::Point2f> points2,
                              const float inlierDistance = -1);


std::vector<std::vector<cv::Point>> GetContours(cv::Mat img, int th = 20, int dilation_size = 3, bool bShow = false);


std::vector<cv::KeyPoint> KeypointsInContour(std::vector<cv::Point> contour, std::vector<cv::KeyPoint> kps);


//std::vector<cv::DMatch> MatchKnn(const cv::Mat &descriptors1, const cv::Mat &descriptors2, float ratio_thresh = 0.8);


cv::Vec3f EquationLine(cv::Point2f p1, cv::Point2f p2);


float line_y_x(cv::Vec3f line, float x);


float line_x_y(cv::Vec3f line, float y);


std::vector<cv::Point3f> FrustumLine(cv::Vec3f line, float lx, float ly);


cv::Vec4f EquationPlane(cv::Point3f p1, cv::Point3f p2, cv::Point3f p3);


cv::Vec4f EquationPlane(cv::Vec3f p1, cv::Vec3f p2, cv::Vec3f p3);


cv::Vec2f LineToLineIntersection(cv::Vec3f l1, cv::Vec3f l2);


float AngleLinePlane(cv::Vec4f pi, cv::Vec3f v);


cv::Vec4f PlaneFromCameraPose(cv::Mat R, cv::Mat t);


cv::Vec3f Intersect2Planes(cv::Vec4f pi1, cv::Vec4f pi2);


float DistancePointLine(const cv::Point2f point, const cv::Vec3f &line);


float DistanceSampson(const cv::Point2f &pt1, const cv::Point2f &pt2, cv::Mat F);


cv::Point3f ptg(cv::Point3f c, cv::Point3f cg, cv::Point2f p, float f);


void DrawCandidates(cv::Mat im1, cv::Mat im2, 
                    cv::Vec3f line, cv::Point2f point, std::vector<cv::Point2f> points, 
                    std::string name = "Candidates");


void DrawCandidates(cv::Mat im1, cv::Mat im2, 
                    cv::Vec3f line, cv::Point2f point, cv::Point2f point2, std::vector<cv::Point2f> points,  
                    std::string name = "Candidates");


void DrawCandidates(cv::Mat im12, 
                    std::vector<cv::Vec3f> line, cv::Point2f point, std::vector<std::vector<cv::Point2f>> points, 
                    std::string name = "Candidates");


cv::Point3f ConvertToWorldCoords(cv::Point2f &p, cv::Mat &R, cv::Mat t, cv::Mat &K);


cv::Point2f ConvertToImageCoords(cv::Point3f &p, cv::Mat &R, cv::Mat t, cv::Mat &K);


cv::Point2f UndistortPointRadial(cv::Point2f &p, cv::Mat &K, cv::Mat &D);


cv::Point2f DistortPointRadial(cv::Point2f &p, cv::Mat &K, cv::Mat &D);


int CountPositive(const std::vector<std::vector<double>> &v);


void PrintPairs(std::vector<cv::DMatch> matches);


std::vector<double> flatten(std::vector<std::vector<double>> &v);

void indices_from_flatten_position(int &i, int &j, int pos, int cols);

std::vector<int> ordered_indices(const std::vector<double> &v);



class ImgLegend {

public:

  ImgLegend(cv::Mat &im, int height, int margin_left, int line_width, int spacing);

  ~ImgLegend();

  void AddLegend(cv::Mat &im, std::string text, cv::Scalar color);

private:
  
  int height_;
  int margin_left_;
  int line_width_;
  int spacing_;

  int num_items_;
  int item_count_;

};


class AngMatcher {

public:

  AngMatcher(std::vector<cv::KeyPoint> vkps1_, std::vector<cv::KeyPoint> vkps2_,
             cv::Mat dsc1_, cv::Mat dsc2_,
             cv::Mat F_, cv::Mat &im1_, cv::Mat &im2_,
             float lx_, float ly_,
             float fo_,
             cv::Point3f co1_, cv::Point3f co2_,
             cv::Point3f co1g_, cv::Point3f co2g_,
             cv::Mat R1_, cv::Mat R2_,
             cv::Mat t_,
             cv::Mat K_);

  ~AngMatcher();

  void Match(std::string method,
             float th_geom, float th_desc, 
             bool bCrossVerification = false, 
             bool draw_inline = false, bool draw_final = false);

  void CompareMatches(std::string method1, std::string method2,
                      int report_level);

  std::vector<std::vector<double>> GetMatches(std::string method);

  std::vector<cv::DMatch> GetMatchesNN(std::string method);

  std::vector<cv::DMatch> GetMatchesDesc(std::string method);

  int MethodMap(std::string method);

  void ViewCandidates(std::vector<std::vector<double>> candidates, int kp, std::string cust_name = "View");

  void ViewCandidates(std::string method, int kp, std::string cust_name = "View");

  void ViewMatches(std::string method, std::string cust_name = "View", float scale = 0.5);

  void ViewKpResults(std::string method1, std::string method2, int kp, std::string cust_name = "View");

  
  

private: 

  // Matches with epipolar line distance and draws the candidates
  std::vector<std::vector<double>> MatchEpilineDist(float th, bool bCrossVerification = false, 
                                                    bool bDraw = false);



  // Matches with angle thresholding and draws the candidates
  std::vector<std::vector<double>> MatchAngle3D(float th, bool bCrossVerification, 
                                                  bool bDraw);


  // Matches Sampson distance
  std::vector<std::vector<double>> MatchSampson(float th, bool bCrossVerification = false, 
                                                bool bDraw = false);

  // Matches with angle thresholding
  std::vector<std::vector<double>> MatchAngle2D(float th, bool bCrossVerification = false, 
                                              bool bDraw = false);


  std::vector<cv::DMatch> MatchDescriptors(std::vector<std::vector<double>> candidates, cv::Mat desc1, cv::Mat desc2, double th);

  std::vector<cv::DMatch> NNCandidates(std::vector<std::vector<double>> candidates, double th);

  std::vector<cv::DMatch> NNCandidates2(std::vector<std::vector<double>> candidates, double th);

  std::vector<int> GetPointIndices(const std::vector<cv::DMatch> &matches1, const std::vector<cv::DMatch> &matches2);





  std::vector<cv::Point2f> kpoints1, kpoints2;
  std::vector<cv::KeyPoint> vkps1, vkps2;
  std::vector<cv::Vec3f> gmlines1, gmlines2;
  cv::Mat dsc1, dsc2;
  cv::Mat F;
  cv::Mat im1, im2;
  float lx, ly;
  float fo;
  cv::Point3f co1, co2;
  cv::Point3f co1g, co2g;
  cv::Mat R1, R2;
  cv::Mat t;
  cv::Mat K;

  std::unordered_map<std::string, std::vector<cv::DMatch>> matches_1_not_2;
  std::unordered_map<std::string, std::vector<cv::DMatch>> matches_2_not_1;
  std::unordered_map<std::string, std::vector<cv::DMatch>> matches_1_and_2;
  std::unordered_map<std::string, std::vector<cv::DMatch>> matches_1_diff_2_1;
  std::unordered_map<std::string, std::vector<cv::DMatch>> matches_1_diff_2_2;


  // Analysis resutls
  unsigned int num_methods_ = 4;
  std::unordered_map<std::string, int> method_map_ = {{"epiline", 0}, {"sampson", 1}, {"angle2d", 2}, {"angle3d", 3}};
  std::vector<std::vector<std::vector<double>>> candidates_;
  std::vector<std::vector<cv::DMatch>> nn_candidates_;
  std::vector<std::vector<cv::DMatch>> desc_matches_;






};






void ResizeAndDisplay(const std::string &title, const cv::Mat &img1, float factor = 1.0, int report_level = 0, bool wait = false, cv::Point2f co = cv::Point2f(0,0));
void ResizeAndDisplay(const std::string &title, const std::vector<cv::Mat> &imgs, float factor);


} // namespace am


#endif