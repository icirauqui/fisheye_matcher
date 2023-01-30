#ifndef FEATURE_MATCHER_H
#define FEATURE_MATCHER_H


#include <iostream>
#include <string>
#include <vector>


#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/viz.hpp>



std::vector<cv::DMatch> MatchKnn(const cv::Mat &descriptors1, const cv::Mat &descriptors2, float ratio_thresh);

std::vector<cv::DMatch> MatchFLANN(const cv::Mat &descriptors1, const cv::Mat &descriptors2, float ratio_thresh);

std::vector<cv::DMatch> MatchBF(const cv::Mat &descriptors1, const cv::Mat &descriptors2, bool crossCheck = false);

cv::Mat CompareMatches(cv::Mat &im1, cv::Mat &im2, 
                       std::vector<cv::KeyPoint> &kps1, std::vector<cv::KeyPoint> &kps2,
                       const std::vector<cv::DMatch> &matches1, const std::vector<cv::DMatch> &matches2, 
                       int report_level = 0);

cv::Mat CompareMatchMethod(cv::Mat &im1, cv::Mat &im2, 
                           std::vector<cv::KeyPoint> &kps1, std::vector<cv::KeyPoint> &kps2,
                           const std::vector<cv::DMatch> &matches1, const std::vector<cv::DMatch> &matches2, 
                           int report_level = 0);


#endif