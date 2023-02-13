#include "feature_matcher.h"




cv::Mat EfromF(const cv::Mat &F, const cv::Mat &K) {
  cv::Mat E = K.t() * F * K;
  //cv::SVD svd(E);
  //cv::Mat W = (cv::Mat_<double>(3, 3) << 0, -1, 0, 1, 0, 0, 0, 0, 1);
  //cv::Mat Wt = (cv::Mat_<double>(3, 3) << 0, 1, 0, -1, 0, 0, 0, 0, 1);
  //E = svd.u * W * svd.vt;
  return E;
}

void RtfromEsvd(const cv::Mat &E, cv::Mat &R, cv::Mat &t) {
  cv::SVD svd(E);
  cv::Mat W = (cv::Mat_<double>(3, 3) << 0, -1, 0, 1, 0, 0, 0, 0, 1);
  cv::Mat Wt = (cv::Mat_<double>(3, 3) << 0, 1, 0, -1, 0, 0, 0, 0, 1);
  R = svd.u * W * svd.vt;
  t = svd.u.col(2);
}

void RtfromE(const cv::Mat &E, cv::Mat &K, cv::Mat &R, cv::Mat &t) {
  cv::recoverPose(E, cv::Mat::zeros(1, 8, CV_64F), cv::Mat::zeros(1, 8, CV_64F), K, R, t);
}


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







FeatureMatcher::FeatureMatcher(cv::Mat im1, cv::Mat im2) {
  im1_ = im1;
  im2_ = im2;
}


FeatureMatcher::~FeatureMatcher() {
}




std::vector<int> FeatureMatcher::GetPointIndices(const std::vector<cv::DMatch> &matches1, const std::vector<cv::DMatch> &matches2) {
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

  return queryIdx;
}


cv::Mat FeatureMatcher::CompareMatches(cv::Mat &im1, cv::Mat &im2, 
                                       std::vector<cv::KeyPoint> &kps1, std::vector<cv::KeyPoint> &kps2,
                                       const std::vector<cv::DMatch> &matches1, const std::vector<cv::DMatch> &matches2, 
                                       int report_level) {

  std::cout << "==== Compare matches ====" << std::endl;

  std::vector<int> queryIdx = GetPointIndices(matches1, matches2);

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


  // Check if targetIdx is different in matches_1_and_2 and store in matches_1_diff_2
  for (unsigned int i=0; i<queryIdx.size(); i++){
    int q = queryIdx[i];
    if (trainIdx1[i] != -1 && trainIdx2[i] != -1){
      auto it1 = std::find_if(matches1.begin(), matches1.end(), [q](cv::DMatch m){return m.queryIdx == q;});
      auto it2 = std::find_if(matches2.begin(), matches2.end(), [q](cv::DMatch m){return m.queryIdx == q;});
      if (it1 != matches2.end() && it2 != matches2.end() && it1->trainIdx != it2->trainIdx){
        matches_1_diff_2_1.push_back(*it1);
        matches_1_diff_2_2.push_back(*it2);
      }
    }
  }

  if (report_level >= 2) {
    // Print matches_1_diff_2
    std::cout << "     Matches Angle and Sampson with different targetIdx" << std::endl;
    std::cout << "      imgIdx\tqueryIdx\ttrainIdx1\ttrainIdx2" << std::endl;
    std::cout << "      ------\t--------\t--------\t--------" << std::endl;
    for (unsigned int i=0; i<matches_1_diff_2_1.size(); i++){
      cv::DMatch m1 = matches_1_diff_2_1[i];
      cv::DMatch m2 = matches_1_diff_2_2[i];
      std::cout << "      " << m1.imgIdx << "\t" << m1.queryIdx << "\t" << m1.trainIdx << "\t" << m2.trainIdx << std::endl;
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







cv::Mat FeatureMatcher::CompareMatchMethod(cv::Mat &im1, cv::Mat &im2, 
                                           std::vector<cv::KeyPoint> &kps1, std::vector<cv::KeyPoint> &kps2,
                                           const std::vector<cv::DMatch> &matches1, const std::vector<cv::DMatch> &matches2, 
                                           int report_level) {

  return cv::Mat();
}





cv::Mat FeatureMatcher::DrawCandidates(cv::Mat &im1, cv::Mat &im2, 
                                    std::vector<cv::KeyPoint> &vkps1, std::vector<cv::KeyPoint> &vkps2,
                                    cv::Mat F,
                                    const std::vector<cv::DMatch> &matches1, const std::vector<cv::DMatch> &matches2, 
                                    int report_level) {
  
  std::cout << " ==== Draw epipolar candidates ==== " << std::endl;
  
  // Get Points from KeyPoints
  std::vector<cv::Point2f> kpoints1, kpoints2;
  for (size_t i = 0; i < vkps1.size(); i++)
    kpoints1.push_back(vkps1[i].pt);
  for (size_t i = 0; i < vkps2.size(); i++)
    kpoints2.push_back(vkps2[i].pt);

  // Compute epilines with given F
  std::vector<cv::Vec3f> gmlines1, gmlines2;
  cv::computeCorrespondEpilines(kpoints1, 1, F, gmlines1);
  cv::computeCorrespondEpilines(kpoints2, 2, F, gmlines2);





  std::vector<int> queryIdx = GetPointIndices(matches1, matches2);
  std::vector<int> trainIdx1, trainIdx2;

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




  std::cout << "Matches 1 not 2 size " << matches_1_not_2.size() << std::endl;
  std::cout << "Matches 2 not 1 size " << matches_2_not_1.size() << std::endl;
  std::cout << "Matches 1 and 2 size " << matches_1_and_2.size() << std::endl;

  if (matches_1_not_2.size() == 0) {
    // Build DMatch vector with matched in sampson but not in angle
    for (unsigned int i=0; i<queryIdx.size(); i++){
      int q = queryIdx[i];
      if (trainIdx1[i] != -1 && trainIdx2[i] == -1){
        auto it = std::find_if(matches1.begin(), matches1.end(), [q](cv::DMatch m){return m.queryIdx == q;});
        if (it != matches1.end()){
          matches_1_not_2.push_back(*it);
        }
      }
    }
  }

  if (matches_2_not_1.size() == 0) {
    // Build DMAtch vector with matched in angle but not in sampson
    for (unsigned int i=0; i<queryIdx.size(); i++){
      int q = queryIdx[i];
      if (trainIdx1[i] == -1 && trainIdx2[i] != -1){
        auto it = std::find_if(matches2.begin(), matches2.end(), [q](cv::DMatch m){return m.queryIdx == q;});
        if (it != matches2.end()){
          matches_2_not_1.push_back(*it);
        }
      }
    }
  }


  if (matches_1_and_2.size() == 0) {
    for (unsigned int i=0; i<queryIdx.size(); i++){
      int q = queryIdx[i];
      if (trainIdx1[i] != -1 && trainIdx2[i] != -1){
        auto it = std::find_if(matches2.begin(), matches2.end(), [q](cv::DMatch m){return m.queryIdx == q;});
        if (it != matches2.end()){
          matches_1_and_2.push_back(*it);
        }
      }
    }
  }


  cv::Mat ims;
  cv::hconcat(im1, im2, ims);


  // For matches_1_not_2
  // Draw point in image 1 and epipolar line in image 2
  for (auto m : matches_1_not_2) {
    cv::Point2f p1 = vkps1[m.queryIdx].pt;
    cv::Point2f p2 = vkps2[m.trainIdx].pt;
    p2.x += im1.cols;

    // Epiline in image 2
    cv::Vec3f l = gmlines1[m.queryIdx];
    // Points for drawing the line in image 2
    cv::Point2f p3, p4;
    p3.x = 0;
    p3.y = -l[2]/l[1];
    p4.x = im2.cols;
    p4.y = -(l[2]+l[0]*im2.cols)/l[1];
    p3.x += im1.cols;
    p4.x += im1.cols;

    // Epiline in image 1
    cv::Vec3f l2 = gmlines2[m.trainIdx];
    // Points for drawing the line in image 1
    cv::Point2f p5, p6;
    p5.x = 0;
    p5.y = -l2[2]/l2[1];
    p6.x = im1.cols;
    p6.y = -(l2[2]+l2[0]*im1.cols)/l2[1];
    
    // Random color
    cv::Scalar color(rand() % 255, rand() % 255, rand() % 255);

    // Draw epipolar lines
    cv::line(ims, p3, p4, color, 3);
    cv::line(ims, p5, p6, color, 3);

    // Draw circle around keypoint
    cv::circle(ims, p1, 3, color, 3);
    cv::circle(ims, p2, 3, color, 3);
  }


/*
  // For matches_2_not_1
  // Draw point in image 2 and epipolar line in image 1

  for (auto m : matches_2_not_1) {
    cv::Point2f p1 = vkps1[m.trainIdx].pt;
    cv::Point2f p2 = vkps2[m.queryIdx].pt;
    p2.x += im1.cols;
    cv::line(ims, p1, p2, cv::Scalar(0,255,0), 1);
    // Draw circle around keypoint
    cv::circle(ims, p1, 3, cv::Scalar(0,255,0), 1);
    cv::circle(ims, p2, 3, cv::Scalar(0,255,0), 1);
  }

  // For matches_1_and_2
  // Draw point in image 1 and epipolar line in image 2
  for (auto m : matches_1_and_2) {
    cv::Point2f p1 = vkps1[m.queryIdx].pt;
    cv::Point2f p2 = vkps2[m.trainIdx].pt;
    p2.x += im1.cols;
    cv::line(ims, p1, p2, cv::Scalar(0,0,255), 1);
    // Draw circle around keypoint
    cv::circle(ims, p1, 3, cv::Scalar(0,0,255), 1);
    cv::circle(ims, p2, 3, cv::Scalar(0,0,255), 1);
  }

*/

  return ims;
}
