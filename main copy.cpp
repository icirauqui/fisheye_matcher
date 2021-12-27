#include <iostream>
#include <math.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

float fx = 717.2104;
float fy = 717.4816;
float cx = 735.3566;
float cy = 552.7982;
cv::Mat K = (cv::Mat_<float>(3,3) << 717.2104, 0, 735.3566, 0, 717.4816, 552.7982, 0, 0, 1);
float k1 = -0.1389272;
float k2 = -0.001239606;
float k3 = 0.0009125824;
float k4 = -0.00004071615;
cv::Mat dist = (cv::Mat_<float>(1,4) << -0.1389272, -0.001239606, 0.0009125824, -0.00004071615);


class Line{
    float a = 0.0;
    float b = 0.0;
    float c = 0.0;
    float d = 0.0;
};


// Function to find equation of plane.
cv::Mat equation_plane(cv::Point3f p1, cv::Point3f p2, cv::Point3f p3){
    float a1 = p2.x - p1.x;
    float b1 = p2.y - p1.y;
    float c1 = p2.z - p1.z;
    float a2 = p3.x - p1.x;
    float b2 = p3.y - p1.y;
    float c2 = p3.z - p1.z;
    float a = b1 * c2 - b2 * c1;
    float b = a2 * c1 - a1 * c2;
    float c = a1 * b2 - b1 * a2;
    float d = (- a * p1.x - b * p1.y - c * p1.z);

    cv::Mat piMat = (cv::Mat_<float>(1,4) << a, b, c, d);
    //cout << "equation of plane is " << a << " x + " << b << " y + " << c << " z + " << d << " = 0." << std::endl;

    return piMat;
}


void resize_and_display(cv::Mat img1, cv::Mat img2){
    cv::Mat out1, out2;
    cv::resize(img1, out1, cv::Size(), 0.5, 0.5);
    cv::resize(img2, out2, cv::Size(), 0.5, 0.5);

    int rows = cv::max(out1.rows, out2.rows);
    int cols = out1.cols + out2.cols;

    cv::Mat out0(rows, cols, out1.type());

    // Copy images in correct position
    out1.copyTo(out0(cv::Rect(0, 0, out1.cols, out1.rows)));
    out2.copyTo(out0(cv::Rect(out1.cols, 0, out2.cols, out2.rows)));

    cv::imshow("c",out0);
}





int main(){

    cv::Mat im1 = imread("images/1.png", cv::IMREAD_COLOR);
    cv::Mat im2 = imread("images/2.png", cv::IMREAD_COLOR);   
    
    float lx = im1.cols;
    float ly = im1.rows;
    float f = (lx/(lx+ly))*fx + (ly+(lx+ly))*fy;


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
    cv::Mat F12 = cv::findFundamentalMat(points1,points2);
    cv::Mat F21 = cv::findFundamentalMat(points2,points1);

    /*
    std::vector<cv::Mat> lines12, lines21;
    for (size_t i=0; i<kps1.size(); i++){
        float data[3] = { kps1[i].pt.x, kps1[i].pt.y, 1.0 };
        cv::Mat pt = cv::Mat(3, 1, F12.type(), data);
        cv::Mat line = F12 * pt;
        std::cout << line << std::endl;
        lines12.push_back(line);
    }
    */

    std::vector<cv::Mat> lines12, lines21;
    for (size_t i=0; i<points1.size(); i++){
        float data[3] = { points1[i].x, points1[i].y, 1.0 };
        cv::Mat pt = cv::Mat(3, 1, F12.type(), data);
        cv::Mat line = F12 * pt;
        std::cout << line << std::endl;
        lines12.push_back(line);
    }



    std::vector<cv::Point> pts1, pts2;
    for (size_t i=0; i<kps1.size(); i++){
        pts1.push_back(kps1[i].pt);
    }
    for (size_t i=0; i<kps2.size(); i++){
        pts2.push_back(kps2[i].pt);
    }
    std::vector<cv::Point3f> lines1, lines2;
    cv::computeCorrespondEpilines(pts1,1,F12,lines1);	
    cv::computeCorrespondEpilines(pts2,1,F21,lines2);	


    
    for (size_t i = 0; i<points1.size(); i++){
        cv::Mat im1o = im1.clone();
        cv::Mat im2o = im2.clone();
        cv::Scalar color(rand()%255,rand()%255,rand()%255);

        cv::circle(im1o,points1[i],6,color,-1);
        //cv::line(im2o, cv::Point(0,-lines1[i].z/lines1[i].y),
        //               cv::Point(im1o.cols,-(lines1[i].z+lines1[i].x*im1o.cols)/lines1[i].y),
        //               color,3);
        cv::circle(im2o,points2[i],6,color,-1);

        std::cout << lines12[i] << std::endl;

        cv::line(im2o, cv::Point(       0,(-lines12[i].at<float>(2))/lines12[i].at<float>(1)),
                       cv::Point(im2.cols,(-lines12[i].at<float>(2) - (lines12[i].at<float>(0)*im2.cols))/lines12[i].at<float>(1)),
                       color,6);


        resize_and_display(im1o,im2o);

        cv::waitKey(0);
    }
    



    /*
    cv::Point3f co2(cx,cy,-f);
    for (size_t i=0; i<lines1.size(); i++){
        cv::Point3f pi1 = cv::Point3f(0,-lines1[i].z/lines1[i].y,0);
        cv::Point3f pi2 = cv::Point3f(im1.cols,-(lines1[i].z+lines1[i].x*im1.cols)/lines1[i].y,0);

        cv::Mat pi = equation_plane(pi1,pi2,co2);

        std::cout << i << std::endl;
    }
    */



    //-- Draw matches
    cv::Mat img_matches;
    cv::drawMatches( im1, kps1, im2, kps2, good_matches, img_matches, cv::Scalar::all(-1),
                 cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    cv::imshow("Good Matches", img_matches );


    cv::waitKey(0);

    return 0;
}

