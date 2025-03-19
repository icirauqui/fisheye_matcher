#ifndef MATCHER_HPP
#define MATCHER_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>

#include <opencv2/opencv.hpp>

#include <boost/math/tools/polynomial.hpp>
#include <boost/math/tools/roots.hpp>

#include "../fe_lens/fe_lens.hpp"


double AngleLinePlane(cv::Vec4d pi, cv::Vec3d v);

double AngleLinePlane(cv::Vec4d pi, cv::Point3d v);

float DistancePointLine(const cv::Point2f point, const cv::Vec3f &line);

cv::Vec4f EquationPlane(cv::Point3f p1, cv::Point3f p2, cv::Point3f p3);

cv::Vec4f EquationPlane(cv::Vec3f p1, cv::Vec3f p2, cv::Vec3f p3);

cv::Vec3f EquationLine(cv::Point2f p1, cv::Point2f p2);

std::vector<double> flatten(std::vector<std::vector<double>> &v);

void indices_from_flatten_position(int &i, int &j, int pos, int cols);

std::vector<int> ordered_indices(const std::vector<double> &v);


class Matcher {

public:

  Matcher();

  void MatchAngle(FisheyeLens* lens, Image* im1, Image* im2, double th);

  void MatchSampson(FisheyeLens* lens, Image* im1, Image* im2, cv::Mat F, double th);

  std::vector<cv::Point2f> SampsonRegion(FisheyeLens* lens, Image* im1, Image* im2, cv::Mat F, int pt_idx, double th);

  std::vector<cv::DMatch> NNMatches(double th);

  std::vector<cv::DMatch> DescMatches(Image* im1, Image* im2, double th);

  cv::Point2f Match2D(Image* im, 
                      FisheyeLens* lens, 
                      std::vector<cv::DMatch> candidates, 
                      int point_id);

  std::vector<cv::Point3f> Match3D(Image* im, 
                                   FisheyeLens* lens, 
                                   std::vector<cv::DMatch> candidates, 
                                   int point_id);

  std::vector<std::vector<cv::Point3f>> candidates_, candidates_crosscheck_;
  std::vector<std::vector<double>> candidates_val_, candidates_val_crosscheck_;


};

#endif // MATCHER_HPP