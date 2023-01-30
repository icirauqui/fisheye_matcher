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



cv::Mat CompareMatches(cv::Mat &im1, cv::Mat &im2, 
                       std::vector<cv::KeyPoint> &kps1, std::vector<cv::KeyPoint> &kps2,
                       const std::vector<cv::DMatch> &matches1, const std::vector<cv::DMatch> &matches2, 
                       int report_level) {

  std::cout << "==== Compare matches ====" << std::endl;
  std::vector<int> queryIdx;

  for (auto m : matches1){
    auto it = std::find(queryIdx.begin(), queryIdx.end(), m.queryIdx);
    if (it == queryIdx.end()){
      queryIdx.push_back(m.queryIdx);
    }
  }

  for (auto m : matches2){
    auto it = std::find(queryIdx.begin(), queryIdx.end(), m.queryIdx);
    if (it == queryIdx.end()){
      queryIdx.push_back(m.queryIdx);
    }
  }

  // Sort queryIdx
  std::sort(queryIdx.begin(), queryIdx.end());
  std::vector<int> trainIdx1, trainIdx2;

  std::cout << " - queryIdx length: " << queryIdx.size() << std::endl;

  for (auto idx : queryIdx) {
    // Find idx in matches1
    auto it = std::find_if(matches1.begin(), matches1.end(), [idx](cv::DMatch m){return m.queryIdx == idx;});
    if (it != matches1.end()){
      trainIdx1.push_back(it->trainIdx);
    } else {
      trainIdx1.push_back(-1);
    }

    // Find idx in matches2
    it = std::find_if(matches2.begin(), matches2.end(), [idx](cv::DMatch m){return m.queryIdx == idx;});
    if (it != matches2.end()){
      trainIdx2.push_back(it->trainIdx);
    } else {
      trainIdx2.push_back(-1);
    }
  }

  if (report_level >= 1) {
    std::cout << "      queryIdx\tSampson\tAngle" << std::endl;
    std::cout << "      --------\t-------\t-----" << std::endl;
    for (int i = 0; i < queryIdx.size(); i++){
      std::cout << "      " << queryIdx[i] << "\t" << trainIdx1[i] << "\t" << trainIdx2[i] << std::endl;
    }
    std::cout << "      --------\t-------\t-----" << std::endl;
    std::cout << "             \t" << matches1.size() << "\t" << matches2.size() << std::endl;
  }

  // Build DMatch vector with matched in sampson but not in angle
  std::vector<cv::DMatch> matches_1_not_2;
  for (unsigned int i=0; i<queryIdx.size(); i++){
    int q = queryIdx[i];
    if (trainIdx1[i] != -1 && trainIdx2[i] == -1){
      auto it = std::find_if(matches1.begin(), matches1.end(), [q](cv::DMatch m){return m.queryIdx == q;});
      if (it != matches1.end()){
        matches_1_not_2.push_back(*it);
      }
    }
  }


  if (report_level >= 2) {
    // Print matches_1_not_2
    std::cout << "     Matches Sampson not Angle" << std::endl;
    std::cout << "      imgIdx\tqueryIdx\ttrainIdx\tdistance" << std::endl;
    std::cout << "      ------\t--------\t--------\t--------" << std::endl;
    for (auto m : matches_1_not_2){
      std::cout << "      " << m.imgIdx << "\t" << m.queryIdx << "\t" << m.trainIdx << "\t" << m.distance << std::endl;
    }
    std::cout << "      ------\t--------\t--------\t--------" << std::endl;
  }


  // Build DMAtch vector with matched in angle but not in sampson
  std::vector<cv::DMatch> matches_2_not_1;
  for (unsigned int i=0; i<queryIdx.size(); i++){
    int q = queryIdx[i];
    if (trainIdx1[i] == -1 && trainIdx2[i] != -1){
      auto it = std::find_if(matches2.begin(), matches2.end(), [q](cv::DMatch m){return m.queryIdx == q;});
      if (it != matches2.end()){
        matches_2_not_1.push_back(*it);
      }
    }
  }


  if (report_level >= 2) {
    // Print matches_2_not_1
    std::cout << "     Matches Angle not Sampson" << std::endl;
    std::cout << "      imgIdx\tqueryIdx\ttrainIdx\tdistance" << std::endl;
    std::cout << "      ------\t--------\t--------\t--------" << std::endl;
    for (auto m : matches_2_not_1){
      std::cout << "      " << m.imgIdx << "\t" << m.queryIdx << "\t" << m.trainIdx << "\t" << m.distance << std::endl;
    }
    std::cout << "      ------\t--------\t--------\t--------" << std::endl;
  }


  // Build DMatch vector with matched in angle and in sampson
  std::vector<cv::DMatch> matches_1_and_2;
  for (unsigned int i=0; i<queryIdx.size(); i++){
    int q = queryIdx[i];
    if (trainIdx1[i] != -1 && trainIdx2[i] != -1){
      auto it = std::find_if(matches2.begin(), matches2.end(), [q](cv::DMatch m){return m.queryIdx == q;});
      if (it != matches2.end()){
        matches_1_and_2.push_back(*it);
      }
    }
  }

  if (report_level >= 2) {
    // Print matches_angle_and_sampson
    std::cout << "     Matches Angle and Sampson" << std::endl;
    std::cout << "      imgIdx\tqueryIdx\ttrainIdx\tdistance" << std::endl;
    std::cout << "      ------\t--------\t--------\t--------" << std::endl;
    for (auto m : matches_1_and_2){
      std::cout << "      " << m.imgIdx << "\t" << m.queryIdx << "\t" << m.trainIdx << "\t" << m.distance << std::endl;
    }
    std::cout << "      ------\t--------\t--------\t--------" << std::endl;
  }




  // Draw segregation in a single image differentiating by color
  // Join image 1 and image 2
  cv::Mat imout_matches_segregation;
  cv::hconcat(im1, im2, imout_matches_segregation);
  
  // Line thickness
  int thickness = 1;
  int radius = 6;
  
  // Draw matches in angle but not in sampson in green
  for (auto m : matches_2_not_1){
    cv::Point2f p1 = kps1[m.queryIdx].pt;
    cv::Point2f p2 = kps2[m.trainIdx].pt;
    p2.x += im1.cols;
    cv::line(imout_matches_segregation, p1, p2, cv::Scalar(0,255,0), thickness);
    // Draw circle around keypoint
    cv::circle(imout_matches_segregation, p1, radius, cv::Scalar(0,255,0), thickness);
    cv::circle(imout_matches_segregation, p2, radius, cv::Scalar(0,255,0), thickness);
  }

  // Draw matches in sampson but not in angle in red
  for (auto m : matches_1_not_2){
    cv::Point2f p1 = kps1[m.queryIdx].pt;
    cv::Point2f p2 = kps2[m.trainIdx].pt;
    p2.x += im1.cols;
    cv::line(imout_matches_segregation, p1, p2, cv::Scalar(0,0,255), thickness);
    // Draw circle around keypoint
    cv::circle(imout_matches_segregation, p1, radius, cv::Scalar(0,0,255), thickness);
    cv::circle(imout_matches_segregation, p2, radius, cv::Scalar(0,0,255), thickness);
  }

  // Draw matches in angle and in sampson in blue
  for (auto m : matches_1_and_2){
    cv::Point2f p1 = kps1[m.queryIdx].pt;
    cv::Point2f p2 = kps2[m.trainIdx].pt;
    p2.x += im1.cols;
    cv::line(imout_matches_segregation, p1, p2, cv::Scalar(255,0,0), thickness);
    // Draw circle around keypoint
    cv::circle(imout_matches_segregation, p1, radius, cv::Scalar(255,0,0), thickness);
    cv::circle(imout_matches_segregation, p2, radius, cv::Scalar(255,0,0), thickness);
  }


  //ResizeAndDisplay("Matches Segregation", imout_matches_segregation, 0.5);

  return imout_matches_segregation;
}





cv::Mat CompareMatchMethod(cv::Mat &im1, cv::Mat &im2, 
                           std::vector<cv::KeyPoint> &kps1, std::vector<cv::KeyPoint> &kps2,
                           const std::vector<cv::DMatch> &matches1, const std::vector<cv::DMatch> &matches2, 
                           int report_level) {

  std::cout << "==== Compare matches ====" << std::endl;
  std::vector<int> queryIdx;

  for (auto m : matches1){
    auto it = std::find(queryIdx.begin(), queryIdx.end(), m.queryIdx);
    if (it == queryIdx.end()){
      queryIdx.push_back(m.queryIdx);
    }
  }

  for (auto m : matches2){
    auto it = std::find(queryIdx.begin(), queryIdx.end(), m.queryIdx);
    if (it == queryIdx.end()){
      queryIdx.push_back(m.queryIdx);
    }
  }

  // Sort queryIdx
  std::sort(queryIdx.begin(), queryIdx.end());
  std::vector<int> trainIdx1, trainIdx2;

  std::cout << " - queryIdx length: " << queryIdx.size() << std::endl;

  for (auto idx : queryIdx) {
    // Find idx in matches1
    auto it = std::find_if(matches1.begin(), matches1.end(), [idx](cv::DMatch m){return m.queryIdx == idx;});
    if (it != matches1.end()){
      trainIdx1.push_back(it->trainIdx);
    } else {
      trainIdx1.push_back(-1);
    }

    // Find idx in matches2
    it = std::find_if(matches2.begin(), matches2.end(), [idx](cv::DMatch m){return m.queryIdx == idx;});
    if (it != matches2.end()){
      trainIdx2.push_back(it->trainIdx);
    } else {
      trainIdx2.push_back(-1);
    }
  }

  if (report_level >= 1) {
    std::cout << "      queryIdx\tSampson\tAngle" << std::endl;
    std::cout << "      --------\t-------\t-----" << std::endl;
    for (int i = 0; i < queryIdx.size(); i++){
      std::cout << "      " << queryIdx[i] << "\t" << trainIdx1[i] << "\t" << trainIdx2[i] << std::endl;
    }
    std::cout << "      --------\t-------\t-----" << std::endl;
    std::cout << "             \t" << matches1.size() << "\t" << matches2.size() << std::endl;
  }

  // Build DMatch vector with matched in sampson but not in angle
  std::vector<cv::DMatch> matches_1_not_2;
  for (unsigned int i=0; i<queryIdx.size(); i++){
    int q = queryIdx[i];
    if (trainIdx1[i] != -1 && trainIdx2[i] == -1){
      auto it = std::find_if(matches1.begin(), matches1.end(), [q](cv::DMatch m){return m.queryIdx == q;});
      if (it != matches1.end()){
        matches_1_not_2.push_back(*it);
      }
    }
  }


  if (report_level >= 2) {
    // Print matches_1_not_2
    std::cout << "     Matches Sampson not Angle" << std::endl;
    std::cout << "      imgIdx\tqueryIdx\ttrainIdx\tdistance" << std::endl;
    std::cout << "      ------\t--------\t--------\t--------" << std::endl;
    for (auto m : matches_1_not_2){
      std::cout << "      " << m.imgIdx << "\t" << m.queryIdx << "\t" << m.trainIdx << "\t" << m.distance << std::endl;
    }
    std::cout << "      ------\t--------\t--------\t--------" << std::endl;
  }


  // Build DMAtch vector with matched in angle but not in sampson
  std::vector<cv::DMatch> matches_2_not_1;
  for (unsigned int i=0; i<queryIdx.size(); i++){
    int q = queryIdx[i];
    if (trainIdx1[i] == -1 && trainIdx2[i] != -1){
      auto it = std::find_if(matches2.begin(), matches2.end(), [q](cv::DMatch m){return m.queryIdx == q;});
      if (it != matches2.end()){
        matches_2_not_1.push_back(*it);
      }
    }
  }


  if (report_level >= 2) {
    // Print matches_2_not_1
    std::cout << "     Matches Angle not Sampson" << std::endl;
    std::cout << "      imgIdx\tqueryIdx\ttrainIdx\tdistance" << std::endl;
    std::cout << "      ------\t--------\t--------\t--------" << std::endl;
    for (auto m : matches_2_not_1){
      std::cout << "      " << m.imgIdx << "\t" << m.queryIdx << "\t" << m.trainIdx << "\t" << m.distance << std::endl;
    }
    std::cout << "      ------\t--------\t--------\t--------" << std::endl;
  }


  // Build DMatch vector with matched in angle and in sampson
  std::vector<cv::DMatch> matches_1_and_2;
  for (unsigned int i=0; i<queryIdx.size(); i++){
    int q = queryIdx[i];
    if (trainIdx1[i] != -1 && trainIdx2[i] != -1){
      auto it = std::find_if(matches2.begin(), matches2.end(), [q](cv::DMatch m){return m.queryIdx == q;});
      if (it != matches2.end()){
        matches_1_and_2.push_back(*it);
      }
    }
  }

  if (report_level >= 2) {
    // Print matches_angle_and_sampson
    std::cout << "     Matches Angle and Sampson" << std::endl;
    std::cout << "      imgIdx\tqueryIdx\ttrainIdx\tdistance" << std::endl;
    std::cout << "      ------\t--------\t--------\t--------" << std::endl;
    for (auto m : matches_1_and_2){
      std::cout << "      " << m.imgIdx << "\t" << m.queryIdx << "\t" << m.trainIdx << "\t" << m.distance << std::endl;
    }
    std::cout << "      ------\t--------\t--------\t--------" << std::endl;
  }




  // Draw segregation in a single image differentiating by color
  // Join image 1 and image 2
  cv::Mat imout_matches_segregation;
  cv::hconcat(im1, im2, imout_matches_segregation);
  
  // Line thickness
  int thickness = 1;
  int radius = 6;
  
  // Draw matches in angle but not in sampson in green
  for (auto m : matches_2_not_1){
    cv::Point2f p1 = kps1[m.queryIdx].pt;
    cv::Point2f p2 = kps2[m.trainIdx].pt;
    p2.x += im1.cols;
    cv::line(imout_matches_segregation, p1, p2, cv::Scalar(0,255,0), thickness);
    // Draw circle around keypoint
    cv::circle(imout_matches_segregation, p1, radius, cv::Scalar(0,255,0), thickness);
    cv::circle(imout_matches_segregation, p2, radius, cv::Scalar(0,255,0), thickness);
  }

  // Draw matches in sampson but not in angle in red
  for (auto m : matches_1_not_2){
    cv::Point2f p1 = kps1[m.queryIdx].pt;
    cv::Point2f p2 = kps2[m.trainIdx].pt;
    p2.x += im1.cols;
    cv::line(imout_matches_segregation, p1, p2, cv::Scalar(0,0,255), thickness);
    // Draw circle around keypoint
    cv::circle(imout_matches_segregation, p1, radius, cv::Scalar(0,0,255), thickness);
    cv::circle(imout_matches_segregation, p2, radius, cv::Scalar(0,0,255), thickness);
  }

  // Draw matches in angle and in sampson in blue
  for (auto m : matches_1_and_2){
    cv::Point2f p1 = kps1[m.queryIdx].pt;
    cv::Point2f p2 = kps2[m.trainIdx].pt;
    p2.x += im1.cols;
    cv::line(imout_matches_segregation, p1, p2, cv::Scalar(255,0,0), thickness);
    // Draw circle around keypoint
    cv::circle(imout_matches_segregation, p1, radius, cv::Scalar(255,0,0), thickness);
    cv::circle(imout_matches_segregation, p2, radius, cv::Scalar(255,0,0), thickness);
  }


  //ResizeAndDisplay("Matches Segregation", imout_matches_segregation, 0.5);

  return imout_matches_segregation;
}

