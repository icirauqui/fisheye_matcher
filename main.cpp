#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
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
double data[9] = {fx, 0, cx, 0, fy, cy, 0, 0, 1};
cv::Mat K = cv::Mat(3, 3, CV_64F, &data);
//cv::Mat K = cv::Mat(3, 1, CV_64FC1, &data);
float k1 = -0.1389272;
float k2 = -0.001239606;
float k3 = 0.0009125824;
float k4 = -0.00004071615;





float distancePointLine(const cv::Point2f point, const cv::Point3f& line){
  //Line is given as a*x + b*y + c = 0
  return std::fabs(line.x*point.x + line.y*point.y + line.z)
      / std::sqrt(line.x*line.x+line.y*line.y);
}


/*
// Function to find equation of plane.
void equation_plane(float x1, float y1, float z1, 
                    float x2, float y2, float z2,
                    float x3, float y3, float z3) {
    float a1 = x2 - x1;
    float b1 = y2 - y1;
    float c1 = z2 - z1;
    float a2 = x3 - x1;
    float b2 = y3 - y1;
    float c2 = z3 - z1;
    float a = b1 * c2 - b2 * c1;
    float b = a2 * c1 - a1 * c2;
    float c = a1 * b2 - b1 * a2;
    float d = (- a * x1 - b * y1 - c * z1);
    cout << "equation of plane is " << a << " x + " << b
        << " y + " << c << " z + " << d << " = 0.";
}
*/

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
    cout << "equation of plane is " << a << " x + " << b
        << " y + " << c << " z + " << d << " = 0." << std::endl;

    return piMat;
}







void resize_and_display(Mat img1, Mat img2){
    Mat out1, out2;
    cv::resize(img1, out1, cv::Size(), 0.5, 0.5);
    cv::resize(img2, out2, cv::Size(), 0.5, 0.5);

    int rows = max(out1.rows, out2.rows);
    int cols = out1.cols + out2.cols;

    Mat out0(rows, cols, out1.type());

    // Copy images in correct position
    out1.copyTo(out0(Rect(0, 0, out1.cols, out1.rows)));
    out2.copyTo(out0(Rect(out1.cols, 0, out2.cols, out2.rows)));

    imshow("c",out0);
}




int main()
{

    cv::Mat im1 = imread("images/1.png", cv::IMREAD_COLOR);;
    cv::Mat im2 = imread("images/2.png", cv::IMREAD_COLOR);;

    Ptr<cv::SIFT> f2d = cv::SIFT::create(1000,4,0.04,10,1.6);
    std::vector<cv::KeyPoint> kps1, kps2;    
    cv::Mat desc1, desc2;     
    f2d->detectAndCompute( im1, noArray(), kps1, desc1);
    f2d->detectAndCompute( im2, noArray(), kps2, desc2);

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


    double data_p1[12] = {1, 0, 0, 0,
                         0, 1, 0, 0,
                         0, 0, 1, 0};
    cv::Mat P1 = cv::Mat(3, 4, CV_64F, &data_p1);


    std::cout << "a" << std::endl;
    std::cout << P1 << std::endl;



    cv::Mat E2 = K.t()*F*K;

    std::cout << "E2 = " << E2 << std::endl;

    cv::Mat U, S, V;
    SVD::compute(E2, U, S, V);
    double tempW[9] = {0, -1, 0,
                       1,  0, 0,
                       0,  0, 1};
    cv::Mat W = cv::Mat(3, 3, CV_64F, &tempW);
    std::cout << "U = " << U << std::endl;
    std::cout << "S = " << S << std::endl;
    std::cout << "V = " << V << std::endl;
    cv::Mat Ra = S * W * V.t();
    std::cout << "Ra = " << Ra << std::endl;
    cv::Mat Rb = S * W.t() * V.t();
    std::cout << "Rb = " << Rb << std::endl;


    double tempt[3] = {0, 0, 1};
    cv::Mat temp1 = cv::Mat(3, 1, CV_64F, &tempt);
    cv::Mat t = S * temp1;

    std::cout << "t = " << t << std::endl;

    cv::Mat P2 = cv::Mat(3, 4, CV_64F, &tempt);
    cv::hconcat(Ra,t,P2);
    std::cout << "P2 = " << P2 << std::endl;

    int N = points1.size();
    cv::Mat pnts3D(1,N,CV_64FC4);
    cv::Mat cam0pnts(1,N,CV_64FC2);
    cv::Mat cam1pnts(1,N,CV_64FC2);

    for (unsigned int i=0; i<N; i++){
        Vec3b pnt1 = cam0pnts.at<Vec3b>(Point(0,i));
        pnt1.val[0] = points1[i].x;
        pnt1.val[1] = points1[i].y;
        Vec3b pnt2 = cam1pnts.at<Vec3b>(Point(0,i));
        pnt2.val[0] = points2[i].x;
        pnt2.val[1] = points2[i].y;
    }

    cv::triangulatePoints(P1,P2,cam0pnts,cam1pnts,pnts3D);

    std::cout << pnts3D.size() << std::endl;
    
    Mat im1o = im1.clone();
    Mat im2o = im2.clone();


    float lx = im1o.cols;
    float ly = im1o.rows;
    float f = (lx/(lx+ly))*fx + (ly+(lx+ly))*fy;

    for (size_t i = 0; i<points1.size(); i++){
        cv::Scalar color(rand()%255,rand()%255,rand()%255);

        cv::line(im2o, 
                 cv::Point(0,-lines1[i].z/lines1[i].y),
                 cv::Point(im1o.cols,-(lines1[i].z+lines1[i].x*im1o.cols)/lines1[i].y),
                 color);
        cv::circle(im1o,points1[i],3,color,-1);

        cv::Point3f pi1 = cv::Point3f(0,-lines1[i].z/lines1[i].y,0);
        cv::Point3f pi2 = cv::Point3f(im1o.cols,-(lines1[i].z+lines1[i].x*im1o.cols)/lines1[i].y,0);
        cv::Point3f pi3 = cv::Point3f(cx,cy,-f);

        Mat plane = equation_plane(pi1,pi2,pi3);

        Mat poly1 = im1o.clone();
        Mat poly2 = im2o.clone();
        std::vector<std::vector<cv::Point> > contours;
        std::vector<cv::Point> contour;
        contour.push_back(cv::Point(0,-lines1[i].z/lines1[i].y));
        contour.push_back(cv::Point(im1o.cols,-(lines1[i].z+lines1[i].x*im1o.cols)/lines1[i].y));
        contour.push_back(cv::Point(cx,cy));
        contours.push_back(contour);
        cv::fillPoly(poly2,contours,color);

        resize_and_display(poly1,poly2);
        //std::cout << lines1[i] << std::endl;

        waitKey(0);




    }

    //imshow("a",im1o);
    //imshow("b",im2o);






    Mat out1, out2;
    cv::resize(im1o, out1, cv::Size(), 0.5, 0.5);
    cv::resize(im2o, out2, cv::Size(), 0.5, 0.5);

    int rows = max(out1.rows, out2.rows);
    int cols = out1.cols + out2.cols;

    Mat out0(rows, cols, out1.type());

    // Copy images in correct position
    out1.copyTo(out0(Rect(0, 0, out1.cols, out1.rows)));
    out2.copyTo(out0(Rect(out1.cols, 0, out2.cols, out2.rows)));

    imshow("c",out0);









    //-- Draw matches
    //Mat img_matches;
    //drawMatches( im1, kps1, im2, kps2, good_matches, img_matches, Scalar::all(-1),
    //             Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    ////-- Show detected matches
    //imshow("Good Matches", img_matches );



    waitKey(0);

    return 0;
}



