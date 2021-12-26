#include <iostream>
#include <math.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

cv::Mat K = (cv::Mat_<float>(3,3) << 717.2104, 0, 735.3566, 0, 717.4816, 552.7982, 0, 0, 1);
cv::Mat dist = (cv::Mat_<float>(1,4) << -0.1389272, -0.001239606, 0.0009125824, -0.00004071615);


int main(){

    cv::Mat im1 = imread("images/1.png", cv::IMREAD_COLOR);
    cv::Mat im2 = imread("images/2.png", cv::IMREAD_COLOR);


    // Detect and compute features
    cv::Ptr<cv::SIFT> f2d = cv::SIFT::create(1000,4,0.04,10,1.6);
    std::vector<cv::KeyPoint> kps1, kps2;    
    cv::Mat desc1, desc2;     
    f2d->detectAndCompute( im1, cv::noArray(), kps1, desc1);
    f2d->detectAndCompute( im2, cv::noArray(), kps2, desc2);


    // Matcher BF/KNN
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<cv::DMatch> > knn_matches;
    matcher->knnMatch( desc1, desc2, knn_matches, 2 );
    const float ratio_thresh = 0.45f;
    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    std::vector<cv::Point2f> points1, points2;
    for (unsigned int i=0; i<good_matches.size(); i++){
        cv::Point2f pt1 = kps1[good_matches[i].queryIdx].pt;
        cv::Point2f pt2 = kps2[good_matches[i].trainIdx].pt;
        points1.push_back(pt1);
        points2.push_back(pt2);
    }




    // Epipolar matching
    // Compute F and epilines
    cv::Mat F = cv::findFundamentalMat(points1,points2);

    std::vector<cv::Point3f> lines1, lines2;

    std::vector<cv::Point> pts1;
    for (size_t i=0; i<kps1.size(); i++){
        pts1.push_back(kps1[i].pt);
    }
    cv::computeCorrespondEpilines(pts1,1,F,lines1);	
    //computeCorrespondEpilines(points2,2,F,lines2);	





    return 0;
}

