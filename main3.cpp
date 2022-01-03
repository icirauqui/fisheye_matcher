#include<opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/viz.hpp>

#include <iostream>
#include <math.h>



static float distanceSampson(const cv::Point2f& pt1, const cv::Point2f& pt2, cv::Mat F){
    cv::Mat pt1w = (cv::Mat_<float>(3,1) << pt1.x, pt1.y, 0.);
    cv::Mat pt2w = (cv::Mat_<float>(3,1) << pt2.x, pt2.y, 0.);

    cv::Mat l1 = F.t()*pt2w;
    cv::Mat l2 = F*pt1w;

    cv::Vec3f l1v(l1.at<float>(0), l1.at<float>(1), l1.at<float>(2));
    cv::Vec3f l2v(l2.at<float>(0), l2.at<float>(1), l2.at<float>(2));

    cv::Mat Mnum = pt2w.t()*F*pt1w;
    float fnum = Mnum.at<float>(0,0);
    float den = l1v(0)*l1v(0) + l1v(1)*l1v(1) + l2v(0)*l2v(0) + l2v(1)*l2v(1);
    float d = sqrt(fnum*fnum/den);

    return d;
}



int main(){
    cv::Point2f pta(1.,2);
    cv::Point2f ptb(2.,3);
    cv::Mat F1 = (cv::Mat_<float>(3,3) << 1., 1., 2., 1., 1., 1., 1., 1., 1.);

    float smp = distanceSampson(pta,ptb,F1);
    std::cout << smp << std::endl;


    cv::waitKey(0);
    return 0;
}