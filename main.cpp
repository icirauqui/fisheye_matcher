#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include "opencv2/xfeatures2d.hpp"
//#include "opencv2/xfeatures2d/nonfree.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;

float fx = 717.2104;
float fy = 717.4816;
float cx = 735.3566;
float cy = 552.7982;
float k1 = -0.1389272;
float k2 = -0.001239606;
float k3 = 0.0009125824;
float k4 = -0.00004071615;



int main()
{
    Mat im1 = imread("images/1.png", IMREAD_COLOR);;
    Mat im2 = imread("images/2.png", IMREAD_COLOR);;

    Ptr<SIFT> f2d = SIFT::create(1000,4,0.04,10,1.6);


    //-- Step 1: Detect the keypoints:
    std::vector<KeyPoint> kps1, kps2;    
    Mat desc1, desc2;     
    f2d->detectAndCompute( im1, noArray(), kps1, desc1);
    f2d->detectAndCompute( im2, noArray(), kps2, desc2);

    //-- Step 2: Matching descriptor vectors with a FLANN based matcher
    // Since SURF is a floating-point descriptor NORM_L2 is used
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch( desc1, desc2, knn_matches, 2 );
    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.45f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    std::vector<Point2f> points1, points2;
    for (unsigned int i=0; i<good_matches.size(); i++){
        Point2f pt1 = kps1[good_matches[i].queryIdx].pt;
        Point2f pt2 = kps2[good_matches[i].trainIdx].pt;
        points1.push_back(pt1);
        points2.push_back(pt2);
    }

    Mat F = findFundamentalMat(points1,points2);


    std::vector<Point3f> lines1, lines2;
    computeCorrespondEpilines(points1,1,F,lines1);	
    computeCorrespondEpilines(points2,2,F,lines2);	

    
    Mat im1o = im1.clone();
    Mat im2o = im2.clone();

    for (size_t i = 0; i<points1.size(); i++){
        cv::Scalar color(rand()%255,rand()%255,rand()%255);

        cv::line(im2o, 
                 cv::Point(0,-lines1[i].z/lines1[i].y),
                 cv::Point(im1o.cols,-(lines1[i].z+lines1[i].x*im1o.cols)/lines1[i].y),
                 color);
        cv::circle(im1o,points1[i],3,color,-1);
    }

    imshow("a",im1o);
    imshow("b",im2o);







    //-- Draw matches
    Mat img_matches;
    drawMatches( im1, kps1, im2, kps2, good_matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    //-- Show detected matches
    imshow("Good Matches", img_matches );



    waitKey(0);

    return 0;
}



float distancePointLine(const cv::Point2f point, const cv::Point3f& line){
  //Line is given as a*x + b*y + c = 0
  return std::fabs(line.x*point.x + line.y*point.y + line.z)
      / std::sqrt(line.x*line.x+line.y*line.y);
}
