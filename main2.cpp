#include<opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>


std::vector<std::vector<cv::Point>> get_contours(cv::Mat img, int th = 20, int dilation_size = 3, bool bShow = false){
    //Contour-retrieval modes: RETR_TREE, RETR_LIST, RETR_EXTERNAL, RETR_CCOMP.
    //Contour-approximation methods: CHAIN_APPROX_NONE, CHAIN_APPROX_SIMPLE.
    cv::Mat img_gray;
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

    // Threshold image
    cv::Mat img_thresh;
    cv::threshold(img_gray, img_thresh, th, 255, cv::THRESH_BINARY);

    int dilation_type = cv::MORPH_CROSS;
    cv::Mat element = cv::getStructuringElement(dilation_type,
                       cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                       cv::Point( dilation_size, dilation_size ) );
    cv::erode(img_thresh,img_thresh,element);

    // dilation_type MORPH_RECT, MORPH_CROSS, MORPH_ELLIPSE
    // Detect contours on the binary image
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(img_thresh, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    // Draw contours on the original image
    if (bShow){
        cv::imshow("Binary mage", img_thresh);
        cv::Mat img_copy = img.clone();
        cv::drawContours(img_copy, contours, -1, cv::Scalar(0, 255, 0), 2);
        cv::imshow("None approximation", img_copy);
    }

    return contours;
}

void keypoints_in_contour(std::vector<cv::Point> contour, std::vector<cv::KeyPoint> &kps, cv::Mat &desc){
    std::vector<cv::KeyPoint> kps_tmp = kps;
    cv::Mat desc_tmp = desc;
    int nKpsIn = 0;
    std::vector<bool> bKpsIn = std::vector<bool>(kps.size(), false);

    for (size_t i=0; i<kps.size(); i++){
        if (cv::pointPolygonTest(contour,kps[i].pt,false) > 0){
            nKpsIn++;
            bKpsIn[i] = true;
        }
    }

    kps.clear();
    desc = cv::Mat(nKpsIn,desc_tmp.cols,desc_tmp.type());
    for (size_t i=0; i<kps_tmp.size(); i++){
        if (bKpsIn[i]){
            kps.push_back(kps_tmp[i]);
            desc.row(kps.size()-1) = desc_tmp.row(i);
        }
    }
}


int main() {
    cv::Mat image = imread("images/1.png", cv::IMREAD_COLOR);
    cv::resize(image, image, cv::Size(), 0.5, 0.5);

    std::vector<std::vector<cv::Point>> contours = get_contours(image,20,3,true);

    cv::Ptr<cv::SIFT> f2d = cv::SIFT::create(1000,4,0.04,10,1.6);
    std::vector<cv::KeyPoint> kps1, kps11;    
    cv::Mat desc1, desc2;     
    f2d->detectAndCompute( image, cv::noArray(), kps1, desc1);

    keypoints_in_contour(contours[0],kps1,desc1);


    cv::Mat im1;
    cv::drawKeypoints(image,kps1,im1);
    cv::imshow("KPs1", im1);

    cv::waitKey(0);
    return 0;
}
