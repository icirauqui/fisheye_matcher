#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

// Function to find equation of plane.
cv::Vec4f equation_plane(cv::Point3f p1, cv::Point3f p2, cv::Point3f p3){
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

    cv::Vec4f piVec(a,b,c,d);
    
    return piVec;
}

void resize_and_display(const std::string& title, const cv::Mat& img1, float factor){
    cv::Mat out1;
    cv::resize(img1, out1, cv::Size(), factor, factor);

    cv::imshow(title,out1);
}
 
static float distancePointLine(const cv::Point2f point, const cv::Vec3f& line)
{
  //Line is given as a*x + b*y + c = 0
  return std::fabs(line(0)*point.x + line(1)*point.y + line(2))
      / std::sqrt(line(0)*line(0)+line(1)*line(1));
}

static void drawEpipolarLines(const std::string& title, const cv::Mat F,
                const cv::Mat& img1, const cv::Mat& img2,
                const std::vector<cv::Point2f> points1,
                const std::vector<cv::Point2f> points2,
                const float inlierDistance = -1) {
  CV_Assert(img1.size() == img2.size() && img1.type() == img2.type());
  cv::Mat outImg(img1.rows, img1.cols*2, CV_8UC3);
  cv::Rect rect1(0,0, img1.cols, img1.rows);
  cv::Rect rect2(img1.cols, 0, img1.cols, img1.rows);

  if (img1.type() == CV_8U){
    cv::cvtColor(img1, outImg(rect1), cv::COLOR_GRAY2BGR);
    cv::cvtColor(img2, outImg(rect2), cv::COLOR_GRAY2BGR);
  }
  else{
    img1.copyTo(outImg(rect1));
    img2.copyTo(outImg(rect2));
  }

  std::vector<cv::Vec3f> epilines1, epilines2;
  cv::computeCorrespondEpilines(points1, 1, F, epilines1); //Index starts with 1
  cv::computeCorrespondEpilines(points2, 2, F, epilines2);
 
  CV_Assert(points1.size() == points2.size() && points2.size() == epilines1.size() && epilines1.size() == epilines2.size());
 
  cv::RNG rng(0);
  for(size_t i=0; i<points1.size(); i++) {
    if(inlierDistance > 0) {
      if(distancePointLine(points1[i], epilines2[i]) > inlierDistance ||
        distancePointLine(points2[i], epilines1[i]) > inlierDistance) {
        //The point match is no inlier
        continue;
      }
    }
    /*
     * Epipolar lines of the 1st point set are drawn in the 2nd image and vice-versa
     */
    cv::Scalar color(rng(256),rng(256),rng(256));
 
    cv::line(outImg(rect2),
      cv::Point(0,-epilines1[i][2]/epilines1[i][1]),
      cv::Point(img1.cols,-(epilines1[i][2]+epilines1[i][0]*img1.cols)/epilines1[i][1]),
      color);
    cv::circle(outImg(rect1), points1[i], 3, color, -1, cv::LINE_AA);
 
    cv::line(outImg(rect1),
      cv::Point(0,-epilines2[i][2]/epilines2[i][1]),
      cv::Point(img2.cols,-(epilines2[i][2]+epilines2[i][0]*img2.cols)/epilines2[i][1]),
      color);
    cv::circle(outImg(rect2), points2[i], 3, color, -1, cv::LINE_AA);
  }
  //cv::imshow(title, outImg);
  resize_and_display(title, outImg, 0.5);
  cv::waitKey(1);
}

static void drawEpipolarLines(const std::string& title, const cv::Mat F,
                const cv::Mat& img1, const cv::Mat& img2,
                const std::vector<cv::Point2f> points1,
                const std::vector<cv::Vec3f> epilines1,
                const std::vector<cv::Point2f> points2,
                const float inlierDistance = -1) {
  CV_Assert(img1.size() == img2.size() && img1.type() == img2.type());
  cv::Mat outImg(img1.rows, img1.cols*2, CV_8UC3);
  cv::Rect rect1(0,0, img1.cols, img1.rows);
  cv::Rect rect2(img1.cols, 0, img1.cols, img1.rows);

  if (img1.type() == CV_8U){
    cv::cvtColor(img1, outImg(rect1), cv::COLOR_GRAY2BGR);
    cv::cvtColor(img2, outImg(rect2), cv::COLOR_GRAY2BGR);
  }
  else{
    img1.copyTo(outImg(rect1));
    img2.copyTo(outImg(rect2));
  }
 
  //CV_Assert(points1.size() == points2.size() && points2.size() == epilines1.size() && epilines1.size() == epilines2.size());
 
  cv::RNG rng(0);

  for (size_t i=0; i<points2.size(); i++){
    cv::circle(outImg(rect2), points2[i], 3, cv::Scalar(50,50,50), -1, cv::LINE_AA);
  }

  for(size_t i=0; i<points1.size(); i++) {
    cv::Scalar color(rng(256),rng(256),rng(256));
 
    cv::line(outImg(rect2),
      cv::Point(0,-epilines1[i][2]/epilines1[i][1]),
      cv::Point(img1.cols,-(epilines1[i][2]+epilines1[i][0]*img1.cols)/epilines1[i][1]),
      color);
    cv::circle(outImg(rect1), points1[i], 3, color, -1, cv::LINE_AA);
  }

  resize_and_display(title, outImg, 0.5);
  cv::waitKey(1);
}









int main(){

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

    drawEpipolarLines("epip1",F12,im1,im2,points1,points2);





    std::vector<cv::Point2f> kpoints1, kpoints2;
    for (size_t i=0; i<kps1.size(); i++)
      kpoints1.push_back(kps1[i].pt);
    for (size_t i=0; i<kps2.size(); i++)
      kpoints2.push_back(kps2[i].pt);

    std::vector<cv::Vec3f> gmlines1, gmlines2;
    //cv::computeCorrespondEpilines(kpoints1, 1, F12, gmlines1);
    //cv::computeCorrespondEpilines(kpoints2, 2, F12, gmlines2);
    cv::computeCorrespondEpilines(points1, 1, F12, gmlines1);
    cv::computeCorrespondEpilines(points2, 2, F12, gmlines2);

    //drawEpipolarLines("epip2",F12,im1,im2,kpoints1,gmlines1,kpoints2);
    drawEpipolarLines("epip2",F12,im1,im2,points1,gmlines1,points2);

    cv::Vec3f c2v(cx,cy,f);
    cv::Point3f c2(cx,cy,f);

    std::vector<cv::Point2f> gmpoints1, gmpoints2;
    std::vector< std::vector<cv::DMatch> > gm_matches;
    for (size_t i=0; i<gmlines1.size(); i++){
      cv::Point3f pt0(0,-gmlines1[i][2]/gmlines1[i][1],0.);
      cv::Point3f pt1(im1.cols,-(gmlines1[i][2]+gmlines1[i][0]*im1.cols)/gmlines1[i][1],0.);
      cv::Vec4f pi = equation_plane(pt0, pt1, c2);

      //for (size_t j=0; j<kpoints2.size(); j++){
      for (size_t j=0; j<points2.size(); j++){
        //cv::Vec3f kp(kpoints2[j].x, kpoints2[j].y, 0.);
        cv::Vec3f kp(points2[j].x, points2[j].y, 0.);

        //Vector director de la recta
        cv::Vec3f v(kp(0)-c2v(0), kp(1)-c2v(1), kp(2)-c2v(2));
        
        //Vector normal del plano
        cv::Vec3f n(pi(1), pi(2), pi(3));

        float num = abs(v(0)*n(0) + v(1)*n(1) + v(2)*n(2));
        float den1 = sqrt(v(0)*v(0) + v(1)*v(1) + v(2)*v(2));
        float den2 = sqrt(n(0)*n(0) + n(1)*n(1) + n(2)*n(2));
        
        float alpha = asin(num / (den1 * den2));

        std::cout << alpha << " ";

      }
      std::cout << std::endl;
    }





    cv::waitKey(0);
    return 0;
}