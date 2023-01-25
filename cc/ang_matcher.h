#ifndef ANG_MATCHER_H
#define ANG_MATCHER_H


#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/viz.hpp>

namespace am {



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


float DistancePointLine(const cv::Point2f point, const cv::Vec3f &line);


float DistanceSampson(const cv::Point2f &pt1, const cv::Point2f &pt2, cv::Mat F);


std::vector<std::vector<double>> MatchSampson(std::vector<cv::KeyPoint> vkps1, std::vector<cv::KeyPoint> vkps2,
                                               cv::Mat dsc1, cv::Mat dsc2,
                                               cv::Mat F,
                                               float lx, float ly, cv::Point3f co2,
                                               float th, bool bCrossVerification = false, 
                                               bool bDraw = false, bool bFiltered = false);


std::vector<std::vector<double>> MatchAngle(std::vector<cv::KeyPoint> vkps1, std::vector<cv::KeyPoint> vkps2,
                                             cv::Mat dsc1, cv::Mat dsc2,
                                             cv::Mat F,
                                             float lx, float ly, cv::Point3f co2,
                                             float th, bool bCrossVerification = false, 
                                             bool bDraw = false, bool bFiltered = false);



std::vector<cv::DMatch> NNCandidates(std::vector<std::vector<double>> candidates, double th);


void HistogramDMatch(const std::string &title, std::vector<cv::DMatch> matches, int th, int factor);


void ResizeAndDisplay(const std::string &title, const cv::Mat &img1, float factor);
void ResizeAndDisplay(const std::string &title, const std::vector<cv::Mat> &imgs, float factor);


} // namespace am


#endif