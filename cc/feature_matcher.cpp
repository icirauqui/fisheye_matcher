#include "feature_matcher.h"







std::vector<cv::DMatch> MatchKnn(const cv::Mat &descriptors1, const cv::Mat &descriptors2, float ratio_thresh) {
  // Match by BF/KNN
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
  std::vector<std::vector<cv::DMatch>> matches_knn_all;
  matcher->knnMatch(descriptors1, descriptors2, matches_knn_all, 2);
  std::vector<cv::DMatch> matches_knn;
  for (size_t i = 0; i < matches_knn_all.size(); i++) {
    if (matches_knn_all[i][0].distance < ratio_thresh * matches_knn_all[i][1].distance) {
      matches_knn.push_back(matches_knn_all[i][0]);
    }
  }
  return matches_knn;
}



std::vector<cv::DMatch> MatchFLANN(const cv::Mat &descriptors1, const cv::Mat &descriptors2, float ratio_thresh) {
  // Match by FLANN
  cv::FlannBasedMatcher matcher;
  std::vector<std::vector<cv::DMatch>> matches_flann_all;
  matcher.knnMatch(descriptors1, descriptors2, matches_flann_all, 2);
  std::vector<cv::DMatch> matches_flann;
  for (size_t i = 0; i < matches_flann_all.size(); i++) {
    if (matches_flann_all[i][0].distance < ratio_thresh * matches_flann_all[i][1].distance) {
      matches_flann.push_back(matches_flann_all[i][0]);
    }
  }
  return matches_flann;
}



std::vector<cv::DMatch> MatchBF(const cv::Mat &descriptors1, const cv::Mat &descriptors2, bool crossCheck) {
  // Match by BF
  cv::BFMatcher matcher(cv::NORM_L2, crossCheck);
  std::vector<cv::DMatch> matches_bf;
  matcher.match(descriptors1, descriptors2, matches_bf);
  return matches_bf;
}



