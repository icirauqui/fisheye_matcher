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

cv::Mat EfromF(const cv::Mat &F, const cv::Mat &K);

void RtfromEsvd(const cv::Mat &E, cv::Mat &R, cv::Mat &t);

void RtfromE(const cv::Mat &E, cv::Mat &K, cv::Mat &R, cv::Mat &t);

std::vector<cv::DMatch> MatchKnn(const cv::Mat &descriptors1, const cv::Mat &descriptors2, float ratio_thresh);

std::vector<cv::DMatch> MatchFLANN(const cv::Mat &descriptors1, const cv::Mat &descriptors2, float ratio_thresh);

std::vector<cv::DMatch> MatchBF(const cv::Mat &descriptors1, const cv::Mat &descriptors2, bool crossCheck = false);


class FeatureMatcher {

public:

  FeatureMatcher(cv::Mat im1, cv::Mat im2);

  ~FeatureMatcher();

  std::vector<int> GetPointIndices(const std::vector<cv::DMatch> &matches1, const std::vector<cv::DMatch> &matches2);


  cv::Mat CompareMatches(cv::Mat &im1, cv::Mat &im2, 
                        std::vector<cv::KeyPoint> &kps1, std::vector<cv::KeyPoint> &kps2,
                        const std::vector<cv::DMatch> &matches1, const std::vector<cv::DMatch> &matches2, 
                        int report_level = 0);

  cv::Mat CompareMatchMethod(cv::Mat &im1, cv::Mat &im2, 
                            std::vector<cv::KeyPoint> &kps1, std::vector<cv::KeyPoint> &kps2,
                            const std::vector<cv::DMatch> &matches1, const std::vector<cv::DMatch> &matches2, 
                            int report_level = 0);

  cv::Mat DrawCandidates(cv::Mat &im1, cv::Mat &im2, 
                      std::vector<cv::KeyPoint> &vkps1, std::vector<cv::KeyPoint> &vkps2,
                      cv::Mat F,
                      const std::vector<cv::DMatch> &matches1, const std::vector<cv::DMatch> &matches2, 
                      int report_level = 0);

      

private:

  cv::Mat im1_;
  cv::Mat im2_;

  std::vector<cv::DMatch> matches_1_not_2;
  std::vector<cv::DMatch> matches_2_not_1;
  std::vector<cv::DMatch> matches_1_and_2;
  std::vector<cv::DMatch> matches_1_diff_2_1;
  std::vector<cv::DMatch> matches_1_diff_2_2;


};


#endif