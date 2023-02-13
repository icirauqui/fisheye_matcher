#include "ang_matcher.h"


namespace am {




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






cv::Mat CompareEpipolarLines(const std::string &title, const cv::Mat F,
                              const cv::Mat &img1, const cv::Mat &img2,
                              const std::vector<cv::Point2f> points1, const std::vector<cv::Point2f> points2,
                              const float inlierDistance) {
  CV_Assert(img1.size() == img2.size() && img1.type() == img2.type());
  cv::Mat outImg(img1.rows, img1.cols * 2, CV_8UC3);
  cv::Rect rect1(0, 0, img1.cols, img1.rows);
  cv::Rect rect2(img1.cols, 0, img1.cols, img1.rows);

  if (img1.type() == CV_8U)
  {
    cv::cvtColor(img1, outImg(rect1), cv::COLOR_GRAY2BGR);
    cv::cvtColor(img2, outImg(rect2), cv::COLOR_GRAY2BGR);
  }
  else
  {
    img1.copyTo(outImg(rect1));
    img2.copyTo(outImg(rect2));
  }

  std::vector<cv::Vec3f> epilines1, epilines2;
  cv::computeCorrespondEpilines(points1, 1, F, epilines1); // Index starts with 1
  cv::computeCorrespondEpilines(points2, 2, F, epilines2);

  //std::cout << "OCV " << epilines1[0][0] << " " << epilines1[0][1] << " " << epilines1[0][2] << std::endl;

  CV_Assert(points1.size() == points2.size() && points2.size() == epilines1.size() && epilines1.size() == epilines2.size());

  for (size_t i = 0; i < points1.size(); i++)
  {
    if (inlierDistance > 0)
    {
      if (DistancePointLine(points1[i], epilines2[i]) > inlierDistance ||
          DistancePointLine(points2[i], epilines1[i]) > inlierDistance)
      {
        // The point match is no inlier
        continue;
      }
    }
    cv::Scalar color(cv::RNG(256), cv::RNG(256), cv::RNG(256));

    cv::line(outImg(rect2),
             cv::Point(0, -epilines1[i][2] / epilines1[i][1]),
             cv::Point(img1.cols, -(epilines1[i][2] + epilines1[i][0] * img1.cols) / epilines1[i][1]),
             color);
    cv::circle(outImg(rect1), points1[i], 3, color, -1, cv::LINE_AA);

    cv::line(outImg(rect1),
             cv::Point(0, -epilines2[i][2] / epilines2[i][1]),
             cv::Point(img2.cols, -(epilines2[i][2] + epilines2[i][0] * img2.cols) / epilines2[i][1]),
             color);
    cv::circle(outImg(rect2), points2[i], 3, color, -1, cv::LINE_AA);
  }

  return outImg;
}


void DrawEpipolarLines(const std::string &title, const cv::Mat F,
                              const cv::Mat &img1, const cv::Mat &img2,
                              const std::vector<cv::Point2f> points1, const std::vector<cv::Point2f> points2,
                              const float inlierDistance) {
  CV_Assert(img1.size() == img2.size() && img1.type() == img2.type());
  cv::Mat outImg(img1.rows, img1.cols * 2, CV_8UC3);
  cv::Rect rect1(0, 0, img1.cols, img1.rows);
  cv::Rect rect2(img1.cols, 0, img1.cols, img1.rows);

  if (img1.type() == CV_8U)
  {
    cv::cvtColor(img1, outImg(rect1), cv::COLOR_GRAY2BGR);
    cv::cvtColor(img2, outImg(rect2), cv::COLOR_GRAY2BGR);
  }
  else
  {
    img1.copyTo(outImg(rect1));
    img2.copyTo(outImg(rect2));
  }

  std::vector<cv::Vec3f> epilines1, epilines2;
  cv::computeCorrespondEpilines(points1, 1, F, epilines1); // Index starts with 1
  cv::computeCorrespondEpilines(points2, 2, F, epilines2);

  CV_Assert(points1.size() == points2.size() && points2.size() == epilines1.size() && epilines1.size() == epilines2.size());

  for (size_t i = 0; i < points1.size(); i++)
  {
    if (inlierDistance > 0)
    {
      if (DistancePointLine(points1[i], epilines2[i]) > inlierDistance ||
          DistancePointLine(points2[i], epilines1[i]) > inlierDistance)
      {
        // The point match is no inlier
        continue;
      }
    }
    cv::Scalar color(cv::RNG(256), cv::RNG(256), cv::RNG(256));

    cv::line(outImg(rect2),
             cv::Point(0, -epilines1[i][2] / epilines1[i][1]),
             cv::Point(img1.cols, -(epilines1[i][2] + epilines1[i][0] * img1.cols) / epilines1[i][1]),
             color);
    cv::circle(outImg(rect1), points1[i], 3, color, -1, cv::LINE_AA);

    cv::line(outImg(rect1),
             cv::Point(0, -epilines2[i][2] / epilines2[i][1]),
             cv::Point(img2.cols, -(epilines2[i][2] + epilines2[i][0] * img2.cols) / epilines2[i][1]),
             color);
    cv::circle(outImg(rect2), points2[i], 3, color, -1, cv::LINE_AA);
  }
  ResizeAndDisplay(title, outImg, 0.5);
  cv::waitKey(0);
}



void DrawEpipolarLines(const std::string &title, const cv::Mat F,
                              const cv::Mat &img1, const cv::Mat &img2,
                              const std::vector<cv::Point2f> points1,
                              const std::vector<cv::Vec3f> epilines1,
                              const std::vector<cv::Point2f> points2,
                              const float inlierDistance) {
  CV_Assert(img1.size() == img2.size() && img1.type() == img2.type());
  cv::Mat outImg(img1.rows, img1.cols * 2, CV_8UC3);
  cv::Rect rect1(0, 0, img1.cols, img1.rows);
  cv::Rect rect2(img1.cols, 0, img1.cols, img1.rows);

  if (img1.type() == CV_8U)
  {
    cv::cvtColor(img1, outImg(rect1), cv::COLOR_GRAY2BGR);
    cv::cvtColor(img2, outImg(rect2), cv::COLOR_GRAY2BGR);
  }
  else
  {
    img1.copyTo(outImg(rect1));
    img2.copyTo(outImg(rect2));
  }

  for (size_t i = 0; i < points2.size(); i++)
  {
    cv::circle(outImg(rect2), points2[i], 3, cv::Scalar(50, 50, 50), -1, cv::LINE_AA);
  }

  for (size_t i = 0; i < points1.size(); i++)
  {
    cv::Scalar color(cv::RNG(256), cv::RNG(256), cv::RNG(256));

    cv::line(outImg(rect2),
             cv::Point(0, -epilines1[i][2] / epilines1[i][1]),
             cv::Point(img1.cols, -(epilines1[i][2] + epilines1[i][0] * img1.cols) / epilines1[i][1]),
             color);
    cv::circle(outImg(rect1), points1[i], 3, color, -1, cv::LINE_AA);
  }

  ResizeAndDisplay(title, outImg, 0.5);
  cv::waitKey(0);
}





std::vector<std::vector<cv::Point>> GetContours(cv::Mat img, int th, int dilation_size, bool bShow) {
  // Contour-retrieval modes: RETR_TREE, RETR_LIST, RETR_EXTERNAL, RETR_CCOMP.
  // Contour-approximation methods: CHAIN_APPROX_NONE, CHAIN_APPROX_SIMPLE.
  cv::Mat img_gray;
  cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

  // Threshold image
  cv::Mat img_thresh;
  cv::threshold(img_gray, img_thresh, th, 255, cv::THRESH_BINARY);

  int dilation_type = cv::MORPH_CROSS;
  cv::Mat element = cv::getStructuringElement(dilation_type,
                                              cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
                                              cv::Point(dilation_size, dilation_size));
  cv::erode(img_thresh, img_thresh, element);

  // dilation_type MORPH_RECT, MORPH_CROSS, MORPH_ELLIPSE
  // Detect contours on the binary image
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(img_thresh, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

  // Draw contours on the original image
  if (bShow) {
    cv::imshow("Binary mage", img_thresh);
    cv::Mat img_copy = img.clone();
    cv::drawContours(img_copy, contours, -1, cv::Scalar(0, 255, 0), 2);
    cv::imshow("None approximation", img_copy);
  }

  return contours;
}


std::vector<cv::KeyPoint> KeypointsInContour(std::vector<cv::Point> contour, std::vector<cv::KeyPoint> kps) {
  std::vector<cv::KeyPoint> kps_tmp;
  for (size_t i = 0; i < kps.size(); i++) {
    if (cv::pointPolygonTest(contour, kps[i].pt, false) > 0) {
      kps_tmp.push_back(kps[i]);
    }
  }
  return kps_tmp;
}


cv::Vec3f EquationLine(cv::Point2f p1, cv::Point2f p2) {
  float dx = p2.x - p1.x;
  float dy = p2.y - p1.y;
  float m = dy / dx;
  float c = p1.y - m * p1.x;

  cv::Vec3f eqLine(m, -1, c);
  return eqLine;
}


float line_y_x(cv::Vec3f line, float x) {
  return (line(0) * x + line(2)) / (-line(1));
}


float line_x_y(cv::Vec3f line, float y) {
  return (line(1) * y + line(2)) / (-line(0));
}


std::vector<cv::Point3f> FrustumLine(cv::Vec3f line, float lx, float ly) {
  cv::Point3f pt0(0., line_y_x(line, 0.), 0.);
  if (pt0.y < 0)
    pt0.y = 0;
  else if (pt0.y > ly)
    pt0.y = ly;
  pt0.x = line_x_y(line, pt0.y);

  cv::Point3f pt1(lx, line_y_x(line, lx), 0.);
  if (pt1.y < 0)
    pt1.y = 0;
  else if (pt1.y > ly)
    pt1.y = ly;
  pt1.x = line_x_y(line, pt1.y);

  std::vector<cv::Point3f> frustum = {pt0, pt1};
  return frustum;
}


cv::Vec4f EquationPlane(cv::Point3f p1, cv::Point3f p2, cv::Point3f p3) {
  float a1 = p2.x - p1.x;
  float b1 = p2.y - p1.y;
  float c1 = p2.z - p1.z;
  float a2 = p3.x - p1.x;
  float b2 = p3.y - p1.y;
  float c2 = p3.z - p1.z;
  float a = b1 * c2 - b2 * c1;
  float b = a2 * c1 - a1 * c2;
  float c = a1 * b2 - b1 * a2;
  float d = (-a * p1.x - b * p1.y - c * p1.z);

  cv::Vec4f piVec(a, b, c, d);
  return piVec;
}


cv::Vec4f EquationPlane(cv::Vec3f p1, cv::Vec3f p2, cv::Vec3f p3) {
  float a1 = p2(0) - p1(0);
  float b1 = p2(1) - p1(1);
  float c1 = p2(2) - p1(2);
  float a2 = p3(0) - p1(0);
  float b2 = p3(1) - p1(1);
  float c2 = p3(2) - p1(2);
  float a = b1 * c2 - b2 * c1;
  float b = a2 * c1 - a1 * c2;
  float c = a1 * b2 - b1 * a2;
  float d = (-a * p1(0) - b * p1(1) - c * p1(2));

  cv::Vec4f piVec(a, b, c, d);
  return piVec;
}


cv::Vec2f LineToLineIntersection(cv::Vec3f l1, cv::Vec3f l2) {
  // ax+by+c=0
  float det = l1(0) * l2(1) + l2(0) * l1(1);

  cv::Vec2f intersect(FLT_MAX, FLT_MAX);
  if (det != 0) {
    intersect(0) = (l2(1) * l1(2) - l1(1) * l2(2)) / det;
    intersect(1) = (l1(0) * l2(2) - l2(0) * l1(2)) / det;
  }

  return intersect;
}



float AngleLinePlane(cv::Vec4f pi, cv::Vec3f v) {
  // Normal vector of plane
  cv::Vec3f n(pi(0), pi(1), pi(2));

  float num = abs(v(0) * n(0) + v(1) * n(1) + v(2) * n(2));
  float den1 = sqrt(v(0) * v(0) + v(1) * v(1) + v(2) * v(2));
  float den2 = sqrt(n(0) * n(0) + n(1) * n(1) + n(2) * n(2));

  float beta = acos(num / (den1 * den2));
  float alpha = (M_PI / 2) - beta;
  return alpha;
}



float DistancePointLine(const cv::Point2f point, const cv::Vec3f &line) {
  return std::fabs(line(0) * point.x + line(1) * point.y + line(2)) / std::sqrt(line(0) * line(0) + line(1) * line(1));
}


// Bad
float DistanceSampson(const cv::Point2f &pt1, const cv::Point2f &pt2, cv::Mat F) {
  float data1[3] = {pt1.x, pt1.y, 0.};
  float data2[3] = {pt2.x, pt2.y, 0.};
  cv::Mat pt1w = cv::Mat(3, 1, F.type(), data1);
  cv::Mat pt2w = cv::Mat(3, 1, F.type(), data2);

  cv::Mat l1 = F.t() * pt2w;
  cv::Mat l2 = F * pt1w;

  cv::Vec3f l1v(l1.at<float>(0), l1.at<float>(1), l1.at<float>(2));
  cv::Vec3f l2v(l2.at<float>(0), l2.at<float>(1), l2.at<float>(2));

  cv::Mat Mnum = pt2w.t() * F * pt1w;
  float fnum = Mnum.at<float>(0, 0);
  float den = l1v(0) * l1v(0) + l1v(1) * l1v(1) + l2v(0) * l2v(0) + l2v(1) * l2v(1);
  float d = sqrt(fnum * fnum / den);

  return d;
}



cv::Point3f ptg(cv::Point3f c, cv::Point3f cg, cv::Point2f p, float f) {
  return cv::Point3f(cg.x + p.x - c.x, cg.y + p.y - c.y, cg.z + f);
}



void DrawCandidates(cv::Mat im1, cv::Mat im2, cv::Vec3f line, cv::Point2f point, std::vector<cv::Point2f> points, std::string name) {
  //Concatenate images
  cv::Mat im12;
  cv::hconcat(im1, im2, im12);

  cv::Point2f pt0(0, -line[2] / line[1]);
  cv::Point2f pt1(im2.cols, -(line[2] + line[0] * im2.cols) / line[1]);
  pt0.x += im1.cols;
  pt1.x += im1.cols;
  cv::line(im12, pt0, pt1, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);

  // Draw point in image 1
  cv::circle(im12, point, 8, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

  for (size_t i = 0; i < points.size(); i++){
    cv::Point2f pt = points[i];
    pt.x += im1.cols;
    cv::circle(im12, pt, 8, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
  }

  ResizeAndDisplay(name, im12, 0.5, true);
  //cv::waitKey(0);

}




cv::Point3f ConvertToWorldCoords(cv::Point2f &p, cv::Mat &R, cv::Mat t, cv::Mat &K) {  
  // Point to homogeneous
  cv::Mat p_hom = (cv::Mat_<float>(3, 1) << p.x, p.y, 1);

  // Normalize point
  cv::Mat p_norm = K.inv() * p_hom;
  p_norm.convertTo(p_norm, R.type());

  // Convert to world coordinates
  cv::Mat p_world = R * p_norm + t;  
  p_world.convertTo(p_world, 5);

  return cv::Point3f(p_world.at<float>(0), p_world.at<float>(1), p_world.at<float>(2));
}



int CountPositive(const std::vector<std::vector<double>> &v) {
  int count = 0;
  for (size_t i = 0; i < v.size(); i++) {
    for (size_t j = 0; j < v[i].size(); j++) {
      if (v[i][j] > 0.0) {
        count++;
      }
    }
  }
  return count;
}



void PrintPairs(std::vector<cv::DMatch> matches) {
  for (auto m : matches){
    std::cout << "      " << m.imgIdx << "\t" << m.queryIdx << "\t" << m.trainIdx << "\t" << m.distance << std::endl;
  }
}



std::vector<double> flatten(std::vector<std::vector<double>> &v) {
  std::vector<double> flat;
  for (size_t i = 0; i < v.size(); i++)
    for (size_t j = 0; j < v[i].size(); j++)
      flat.push_back(v[i][j]);
  return flat;
}

void indices_from_flatten_position(int &i, int &j, int pos, int cols) {
  i = pos / cols;
  j = pos % cols;
}

std::vector<int> ordered_indices(const std::vector<double> &v) {
  std::vector<int> index(v.size());
  std::iota(index.begin(), index.end(), 0);
  std::sort(index.begin(), index.end(),
            [&v](int i, int j) { return v[i] < v[j]; });
  return index;
}




ImgLegend::ImgLegend(cv::Mat &im, int height, int margin_left, int line_width, int spacing) {
  height_ = height;
  margin_left_ = margin_left;
  line_width_ = line_width;
  spacing_ = spacing;
  num_items_ = (height_ / spacing_) - 1;

  cv::Mat im_legend;
  im_legend = cv::Mat::zeros(height_, im.cols, CV_8UC3);
  im_legend = cv::Scalar(255,255,255);
  cv::vconcat(im, im_legend, im);
}


ImgLegend::~ImgLegend() {

}


void ImgLegend::AddLegend(cv::Mat &im, std::string text, cv::Scalar color) {
  cv::Point2f pt1(margin_left_, im.rows-((num_items_ - item_count_)*spacing_));
  cv::Point2f pt2(pt1.x + line_width_, pt1.y);
  cv::line(im, pt1, pt2, color, 2);
  cv::Point pt_txt(2*margin_left_ + line_width_, pt1.y+(spacing_/4));
  item_count_++;
  cv::putText(im, text, pt_txt, cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0,0,0), 1);
}





AngMatcher::AngMatcher(std::vector<cv::KeyPoint> vkps1_, std::vector<cv::KeyPoint> vkps2_,
                       cv::Mat dsc1_, cv::Mat dsc2_,
                       cv::Mat F_, cv::Mat &im1_, cv::Mat &im2_,
                       float lx_, float ly_, 
                       float fo_,
                       cv::Point3f co1_, cv::Point3f co2_,
                       cv::Point3f co1g_, cv::Point3f co2g_,
                       cv::Mat R1_, cv::Mat R2_,
                       cv::Mat t_,
                       cv::Mat K_) {
  vkps1 = vkps1_;
  vkps2 = vkps2_;
  dsc1 = dsc1_;
  dsc2 = dsc2_;
  F = F_;
  im1 = im1_;
  im2 = im2_;
  lx = lx_;
  ly = ly_;
  fo = fo_;
  co1 = co1_;
  co2 = co2_;
  co1g = co1g_;
  co2g = co2g_;
  R1 = R1_;
  R2 = R2_;
  t = t_;
  K = K_;

  // Get Points from KeyPoints
  for (size_t i = 0; i < vkps1.size(); i++)
    kpoints1.push_back(vkps1[i].pt);
  for (size_t i = 0; i < vkps2.size(); i++)
    kpoints2.push_back(vkps2[i].pt);

  // Compute epilines with given F
  cv::computeCorrespondEpilines(kpoints1, 1, F, gmlines1);
  cv::computeCorrespondEpilines(kpoints2, 2, F, gmlines2);


  // Initialize data structures for num_methods_
  for (unsigned int i=0; i<num_methods_; i++) {
    for (unsigned int j=i; j<num_methods_; j++) {
      std::string method_key = std::to_string(i) + std::to_string(j);
      matches_1_not_2[method_key] = std::vector<cv::DMatch>();
      matches_2_not_1[method_key] = std::vector<cv::DMatch>();
      matches_1_and_2[method_key] = std::vector<cv::DMatch>();
      matches_1_diff_2_1[method_key] = std::vector<cv::DMatch>();
      matches_1_diff_2_2[method_key] = std::vector<cv::DMatch>();
    }
  }

  //matches_1_not_2 = std::vector<std::vector<cv::DMatch>>(num_methods_);
  //matches_2_not_1 = std::vector<std::vector<cv::DMatch>>(num_methods_);
  //matches_1_and_2 = std::vector<std::vector<cv::DMatch>>(num_methods_);
  //matches_1_diff_2_1 = std::vector<std::vector<cv::DMatch>>(num_methods_);
  //matches_1_diff_2_2 = std::vector<std::vector<cv::DMatch>>(num_methods_);

  candidates_ = std::vector<std::vector<std::vector<double>>>(num_methods_);
  nn_candidates_ = std::vector<std::vector<cv::DMatch>>(num_methods_);
  desc_matches_ = std::vector<std::vector<cv::DMatch>>(num_methods_);

  // Print info
  std::cout << "    AngMatcher initialized. Guided matching methods:" << std::endl
            << "      - Epiline distance" << std::endl
            << "      - Sampson distance" << std::endl
            << "      - Angle 2D" << std::endl
            << "      - Angle 3D" << std::endl << std::endl;
}


AngMatcher::~AngMatcher() {
  //delete this;
}


void AngMatcher::Match(std::string method,
                       float th_geom, float th_desc,
                       bool bCrossVerification, 
                       bool draw_inline, bool draw_final) {

  if (method == "epiline") {
    candidates_[method_map_[method]] = MatchEpilineDist(th_geom, bCrossVerification, draw_inline);
  }
  else if (method == "sampson") {
    candidates_[method_map_[method]] = MatchSampson(th_geom, bCrossVerification, draw_inline);
  }
  else if (method == "angle2d") {
    candidates_[method_map_[method]] = MatchAngle2D(th_geom, bCrossVerification, draw_inline);
  }
  else if (method == "angle3d") {
    candidates_[method_map_[method]] = MatchAngle3D(th_geom, bCrossVerification, draw_inline);
  }
  else {
    std::cout << "Match method doesn't exist" << std::endl;
  }

  if (draw_final) {
    ViewCandidates(candidates_[method_map_[method]], CountMaxIdx(candidates_[method_map_[method]]), method);
  }

  nn_candidates_[method_map_[method]] = NNCandidates(candidates_[method_map_[method]], th_desc);

  desc_matches_[method_map_[method]] = MatchDescriptors(candidates_[method_map_[method]], dsc1, dsc2, th_desc);

  std::cout << " 5." << method_map_[method] + 1 << ". " << method << " all/nn/desc: " 
            << CountPositive(candidates_[method_map_[method]]) << " / " 
            << nn_candidates_[method_map_[method]].size() << " / "
            << desc_matches_[method_map_[method]].size() << std::endl;
}



void AngMatcher::CompareMatches(std::string method1, std::string method2,
                                int report_level) {

  std::vector<cv::KeyPoint> kps1 = vkps1;
  std::vector<cv::KeyPoint> kps2 = vkps2;
  std::vector<cv::DMatch> matches1 = desc_matches_[method_map_[method1]];
  std::vector<cv::DMatch> matches2 = desc_matches_[method_map_[method2]];

  int method1_idx = method_map_[method1];
  int method2_idx = method_map_[method2];
  if (method1_idx > method2_idx) {
    std::cout << "    Swapping methods" << std::endl;
    std::swap(method1_idx, method2_idx);
    std::swap(method1, method2);
    std::swap(matches1, matches2);
  }
  std::string method_key = std::to_string(method1_idx) + std::to_string(method2_idx);

  std::vector<int> queryIdx = GetPointIndices(matches1, matches2);

  std::vector<int> trainIdx1, trainIdx2;

  //std::cout << " - queryIdx length: " << queryIdx.size() << std::endl;
  std::cout << "   =============================" << std::endl;
  std::cout << "    Compare matches " << method1 << " vs " << method2 << std::endl;

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
    std::cout << "      queryIdx\t" << method1 << "\t" << method2 << std::endl;
    std::cout << "      --------\t-------\t-----" << std::endl;
    for (size_t i = 0; i < queryIdx.size(); i++){
      std::cout << "      " << queryIdx[i] << "\t" << trainIdx1[i] << "\t" << trainIdx2[i] << std::endl;
    }
    std::cout << "      --------\t-------\t-----" << std::endl;
    std::cout << "             \t" << matches1.size() << "\t" << matches2.size() << std::endl;
    std::cout << "      --------\t-------\t-----" << std::endl;
  }

  // Build DMatch vector with matched in sampson but not in angle
  for (unsigned int i=0; i<queryIdx.size(); i++){
    int q = queryIdx[i];
    if (trainIdx1[i] != -1 && trainIdx2[i] == -1){
      auto it = std::find_if(matches1.begin(), matches1.end(), [q](cv::DMatch m){return m.queryIdx == q;});
      if (it != matches1.end()){
        matches_1_not_2[method_key].push_back(*it);
      }
    }
  }


  if (report_level >= 2) {
    // Print matches_1_not_2
    std::cout << "     Matches " << method1 << " not " << method2 << std::endl;
    std::cout << "      imgIdx\tqueryIdx\ttrainIdx\tdistance" << std::endl;
    std::cout << "      ------\t--------\t--------\t--------" << std::endl;
    for (auto m : matches_1_not_2[method_key]){
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
        matches_2_not_1[method_key].push_back(*it);
      }
    }
  }


  if (report_level >= 2) {
    // Print matches_2_not_1
    std::cout << "     Matches " << method2 << " not " << method1 << std::endl;
    std::cout << "      imgIdx\tqueryIdx\ttrainIdx\tdistance" << std::endl;
    std::cout << "      ------\t--------\t--------\t--------" << std::endl;
    for (auto m : matches_2_not_1[method_key]){
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
        matches_1_and_2[method_key].push_back(*it);
      }
    }
  }

  if (report_level >= 2) {
    // Print matches_angle_and_sampson
    std::cout << "     Matches " << method1 << " and " << method2 << std::endl;
    std::cout << "      imgIdx\tqueryIdx\ttrainIdx\tdistance" << std::endl;
    std::cout << "      ------\t--------\t--------\t--------" << std::endl;
    for (auto m : matches_1_and_2[method_key]){
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
        matches_1_diff_2_1[method_key].push_back(*it1);
        matches_1_diff_2_2[method_key].push_back(*it2);
      }
    }
  }

  if (report_level >= 2) {
    // Print matches_1_diff_2
    std::cout << "     Matches " << method1 << " and " << method2 << " with different targetIdx" << std::endl;
    std::cout << "      imgIdx\tqueryIdx\ttrainIdx1\ttrainIdx2" << std::endl;
    std::cout << "      ------\t--------\t--------\t--------" << std::endl;
    for (unsigned int i=0; i<matches_1_diff_2_1[method_key].size(); i++){
      cv::DMatch m1 = matches_1_diff_2_1[method_key][i];
      cv::DMatch m2 = matches_1_diff_2_2[method_key][i];
      std::cout << "      " << m1.imgIdx << "\t" << m1.queryIdx << "\t" << m1.trainIdx << "\t" << m2.trainIdx << std::endl;
    }
    std::cout << "      ------\t--------\t--------\t--------" << std::endl;
  }




  std::cout << "    Matches 1 not 2 size  " << matches_1_not_2[method_key].size() << std::endl;
  std::cout << "    Matches 2 not 1 size  " << matches_2_not_1[method_key].size() << std::endl;
  std::cout << "    Matches 1 and 2 size  " << matches_1_and_2[method_key].size() << std::endl;
  std::cout << "    Matches 1 diff 2 size " << matches_1_diff_2_1[method_key].size() << std::endl;




  // Draw segregation in a single image differentiating by color
  // Join image 1 and image 2
  cv::Mat imout_matches_segregation;
  cv::hconcat(im1, im2, imout_matches_segregation);
  
  // Line thickness
  int thickness = 1;
  int radius = 6;
  
  // Draw matches in 1 but not in 2 in red
  for (auto m : matches_1_not_2[method_key]){
    cv::Point2f p1 = kps1[m.queryIdx].pt;
    cv::Point2f p2 = kps2[m.trainIdx].pt;
    p2.x += im1.cols;
    cv::line(imout_matches_segregation, p1, p2, cv::Scalar(0,0,255), thickness);
    // Draw circle around keypoint
    cv::circle(imout_matches_segregation, p1, radius, cv::Scalar(0,0,255), thickness);
    cv::circle(imout_matches_segregation, p2, radius, cv::Scalar(0,0,255), thickness);
  }

  // Draw matches in 2 but not in 1 in green
  for (auto m : matches_2_not_1[method_key]){
    cv::Point2f p1 = kps1[m.queryIdx].pt;
    cv::Point2f p2 = kps2[m.trainIdx].pt;
    p2.x += im1.cols;
    cv::line(imout_matches_segregation, p1, p2, cv::Scalar(0,255,0), thickness);
    // Draw circle around keypoint
    cv::circle(imout_matches_segregation, p1, radius, cv::Scalar(0,255,0), thickness);
    cv::circle(imout_matches_segregation, p2, radius, cv::Scalar(0,255,0), thickness);
  }

  // Draw matches in 1 and in 2 in blue
  for (auto m : matches_1_and_2[method_key]){
    cv::Point2f p1 = kps1[m.queryIdx].pt;
    cv::Point2f p2 = kps2[m.trainIdx].pt;
    p2.x += im1.cols;
    cv::line(imout_matches_segregation, p1, p2, cv::Scalar(255,0,0), thickness);
    // Draw circle around keypoint
    cv::circle(imout_matches_segregation, p1, radius, cv::Scalar(255,0,0), thickness);
    cv::circle(imout_matches_segregation, p2, radius, cv::Scalar(255,0,0), thickness);
  }

  // Draw matches in 1 and in 2 with different targetIdx in black
  for (unsigned int i=0; i<matches_1_diff_2_1[method_key].size(); i++){
    cv::DMatch m1 = matches_1_diff_2_1[method_key][i];
    cv::DMatch m2 = matches_1_diff_2_2[method_key][i];
    cv::Point2f p1 = kps1[m1.queryIdx].pt;
    cv::Point2f p2 = kps2[m1.trainIdx].pt;
    cv::Point2f p3 = kps2[m2.trainIdx].pt;
    p2.x += im1.cols;
    p3.x += im1.cols;
    cv::line(imout_matches_segregation, p1, p2, cv::Scalar(0,0,0), thickness);
    cv::line(imout_matches_segregation, p1, p3, cv::Scalar(0,0,0), thickness);
    // Draw circle around keypoint
    cv::circle(imout_matches_segregation, p1, radius, cv::Scalar(0,0,0), thickness);
    cv::circle(imout_matches_segregation, p2, radius, cv::Scalar(0,0,0), thickness);
    cv::circle(imout_matches_segregation, p3, radius, cv::Scalar(0,0,0), thickness);
  }

  
  // Add legend to image
  ImgLegend legend(imout_matches_segregation, 110, 20, 80, 20);
  legend.AddLegend(imout_matches_segregation, std::to_string(matches_1_not_2[method_key].size()) + " Matches in " + method1 + " but not in " + method2, cv::Scalar(0,0,255));
  legend.AddLegend(imout_matches_segregation, std::to_string(matches_2_not_1[method_key].size()) + " Matches in " + method2 + " but not in " + method1, cv::Scalar(0,255,0));
  legend.AddLegend(imout_matches_segregation, std::to_string(matches_1_and_2[method_key].size()) + " Matches in " + method1 + " and " + method2, cv::Scalar(255,0,0));
  legend.AddLegend(imout_matches_segregation, std::to_string(matches_1_diff_2_1[method_key].size()) + " Matches in " + method1 + " and " + method2 + " with different targetIdx", cv::Scalar(0,0,0));



  std::string frame_title = "Matches Segregation: " + method1 + " vs " + method2;
  ResizeAndDisplay(frame_title, imout_matches_segregation, 0.5);
}


std::vector<std::vector<double>> AngMatcher::GetMatches(std::string method){
  return candidates_[method_map_[method]];
}


std::vector<cv::DMatch> AngMatcher::GetMatchesNN(std::string method){
  return nn_candidates_[method_map_[method]];
}


std::vector<cv::DMatch> AngMatcher::GetMatchesDesc(std::string method){
  return desc_matches_[method_map_[method]];
}


void AngMatcher::ViewCandidates(std::vector<std::vector<double>> candidates, int kp, std::string cust_name) {
  cv::Point3f pt0(0, -gmlines1[kp][2] / gmlines1[kp][1], 0.);
  cv::Point3f pt1(lx, -(gmlines1[kp][2] + gmlines1[kp][0] * lx) / gmlines1[kp][1], 0.);
  cv::Vec3f line = EquationLine(cv::Point2f(pt0.x, pt0.y), cv::Point2f(pt1.x, pt1.y));

  std::vector<cv::Point2f> points;
  for (size_t i = 0; i < candidates[kp].size(); i++) {
    if (candidates[kp][i] > 0.0) {
      cv::Point2f pt(kpoints2[i].x, kpoints2[i].y);
      points.push_back(pt);
    }
  }

  std::string name = cust_name + " " + std::to_string(kp + 1) + " / " + std::to_string(vkps1.size()) + " - " + std::to_string(points.size()) + " candidates";
  DrawCandidates(im1, im2, line, kpoints1[kp], points, name);
}


void AngMatcher::ViewMatches(std::string method, std::string cust_name, float scale) {
  cv::Mat im_matches; 

  cv::drawMatches(im1, vkps1, im2, vkps2, desc_matches_[method_map_[method]], im_matches);
  
  ResizeAndDisplay(cust_name, im_matches, scale);
}



std::vector<std::vector<double>> AngMatcher::MatchEpilineDist(float th, bool bCrossVerification, 
                                                              bool bDraw) {

  // Look for match candidates
  std::vector<std::vector<double>> candidates = std::vector<std::vector<double>>(vkps1.size(), std::vector<double>(vkps2.size(), -1.));

  for (size_t i = 0; i < kpoints1.size(); i++) {
    cv::Point3f pt0(0, -gmlines1[i][2] / gmlines1[i][1], 0.);
    cv::Point3f pt1(lx, -(gmlines1[i][2] + gmlines1[i][0] * lx) / gmlines1[i][1], 0.);
    cv::Vec3f line = EquationLine(cv::Point2f(pt0.x, pt0.y), cv::Point2f(pt1.x, pt1.y));

    std::vector<cv::Point2f> points;

    for (size_t j = 0; j < kpoints2.size(); j++) {
      cv::Point2f kp(kpoints2[j].x, kpoints2[j].y);

      if (DistancePointLine(kp,line) <= th) {
        //double dist_l2 = 0.;
        //if (vkps1.size() == dsc1.rows && vkps2.size() == dsc2.rows)
        //  dist_l2 = norm(dsc1.row(i), dsc2.row(j), cv::NORM_L2);
        double dist_l2 = norm(dsc1.row(i), dsc2.row(j), cv::NORM_L2);
        candidates[i][j] = dist_l2;

        // Cross verification with image 1
        if (bCrossVerification) {
          cv::Point3f im1pt0(0,-gmlines2[j][2]/gmlines2[j][1],0.);
          cv::Point3f im1pt1(lx,-(gmlines2[j][2]+gmlines2[j][0]*lx)/gmlines2[j][1],0.);
          cv::Vec3f im1line = EquationLine(cv::Point2f(im1pt0.x,im1pt0.y), cv::Point2f(im1pt1.x,im1pt1.y));

          cv::Point2f im1kp(kpoints1[i].x, kpoints1[i].y);

          if (DistancePointLine(im1kp,im1line) <= th || !bCrossVerification) {
            //double dist_l2 = 0.;
            //if (vkps1.size() == dsc1.rows && vkps2.size() == dsc2.rows)
            //  dist_l2  = norm(dsc1.row(i),dsc2.row(j),cv::NORM_L2);
            double dist_l2  = norm(dsc1.row(i),dsc2.row(j),cv::NORM_L2);
            candidates[i][j] += dist_l2;

            points.push_back(kp);
          }
        }
      }
    }

    if (points.size() > 0 && bDraw) {
      std::string name = "Epiline " + std::to_string(i + 1) + " / " + std::to_string(vkps1.size()) + " - " + std::to_string(points.size()) + " candidates";
      DrawCandidates(im1, im2, line, kpoints1[i], points, name);
    }
  }

  return candidates;
}



std::vector<std::vector<double>> AngMatcher::MatchAngle3D(float th, bool bCrossVerification, 
                                                        bool bDraw) {


  
  // Look for match candidates
  std::vector<std::vector<double>> candidates = std::vector<std::vector<double>>(vkps1.size(), std::vector<double>(vkps2.size(), -1.));



  for (size_t i = 0; i < vkps1.size(); i++) {
    //cv::Point3f p1g = ptg(co1, co1g, kpoints1[i], fo);

    std::vector<cv::Point2f> points;

    cv::Mat o = cv::Mat::zeros(3,1,t.type());
    cv::Point3f p1g = ConvertToWorldCoords(kpoints1[i], R1, o, K);

    // Epipolar line in image 2: 
    // correct would be to compute the intersection between the plane and the camera 2 plane
    cv::Point3f pt0(0, -gmlines1[i][2] / gmlines1[i][1], 0.);
    cv::Point3f pt1(lx, -(gmlines1[i][2] + gmlines1[i][0] * lx) / gmlines1[i][1], 0.);
    cv::Vec3f line = EquationLine(cv::Point2f(pt0.x, pt0.y), cv::Point2f(pt1.x, pt1.y));

    cv::Vec4f pi = EquationPlane(co1g, co2g, p1g);


    for (size_t j = 0; j < vkps2.size(); j++) {
      cv::Point3f p2g = ConvertToWorldCoords(kpoints2[j], R2, t, K);
      cv::Vec3f v2g(p2g.x - co2g.x, p2g.y - co2g.y, p2g.z - co2g.z);

      float a12 = AngleLinePlane(pi, v2g);
      if (a12 <= th) {
        //candidates[i][j] = a12;

        // Cross verification with image 1
        cv::Vec4f pi2 = EquationPlane(co1g, co2g, p2g);
        cv::Vec3f v1g(p1g.x - co1g.x, p1g.y - co1g.y, p1g.z - co1g.z);

        float a21 = AngleLinePlane(pi2, v1g);
        if (a21 <= th) {
          candidates[i][j] = a12 + a21;
          points.push_back(kpoints2[j]);
        }
      }
    }

    /*
    if (bDraw) {
      cv::viz::Viz3d myWindow("Coordinate Frame");

      // Coordinate system
      myWindow.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem(200));

      // Camera frame
      std::vector<cv::Point3f> camFr = {cv::Point3f(0., ly, 0.), cv::Point3f(lx, ly, 0.), cv::Point3f(lx, 0., 0.), cv::Point3f(0., 0., 0.), cv::Point3f(0., ly, 0.)};
      cv::viz::WPolyLine camFrPoly(camFr, cv::viz::Color::gray());
      myWindow.showWidget("camFrPoly", camFrPoly);

      // Epiplane
      cv::Vec3f line = EquationLine(cv::Point2f(pt0.x, pt0.y), cv::Point2f(pt1.x, pt1.y));
      std::vector<cv::Point3f> lineFr = FrustumLine(line, lx, ly);
      lineFr.push_back(co2);
      lineFr.push_back(lineFr[0]);
      cv::viz::WPolyLine epiplane(lineFr, cv::viz::Color::green());
      myWindow.showWidget("epiplane", epiplane);

      // Candidate points projective rays
      for (size_t j = 0; j < candidates[i].size(); j++) {
        if (candidates[i][j] >= 0.) {
          cv::Point3f kp(kpoints2[j].x, kpoints2[j].y, 0.);
          cv::viz::WLine ptLine(co2, kp, cv::viz::Color::red());
          myWindow.showWidget("ptLine" + j, ptLine);
        }
      }
      if (bFound)
        myWindow.spin();
    }
    */

    
    if (points.size() > 0 && bDraw) {
      std::string name = "Epiline " + std::to_string(i + 1) + " / " + std::to_string(vkps1.size()) + " - " + std::to_string(points.size()) + " candidates";
      DrawCandidates(im1, im2, line, kpoints1[i], points, name);
    }
  }

  return candidates;
}







std::vector<std::vector<double>> AngMatcher::MatchSampson(float th, bool bCrossVerification, 
                                                          bool bDraw) {

  // Look for match candidates
  std::vector<std::vector<double>> candidates = std::vector<std::vector<double>>(vkps1.size(), std::vector<double>(vkps2.size(), -1.));

  for (size_t i = 0; i < vkps1.size(); i++) {
    cv::Point3f pt0(0, -gmlines1[i][2] / gmlines1[i][1], 0.);
    cv::Point3f pt1(lx, -(gmlines1[i][2] + gmlines1[i][0] * lx) / gmlines1[i][1], 0.);
    cv::Vec3f line = EquationLine(cv::Point2f(pt0.x, pt0.y), cv::Point2f(pt1.x, pt1.y));

    cv::Mat point1m(cv::Point3d(vkps1[i].pt.x, vkps1[i].pt.y, 1.0));

    std::vector<cv::Point2f> points;

    for (size_t j = 0; j < kpoints2.size(); j++) {
      cv::Point2f kp(kpoints2[j].x, kpoints2[j].y);

      cv::Mat point2m(cv::Point3d(vkps2[j].pt.x, vkps2[j].pt.y, 1.0));

      if (cv::sampsonDistance(point1m, point2m, F) <= th) {
        //double dist_l2 = 0.;
        //if (vkps1.size() == dsc1.rows && vkps2.size() == dsc2.rows)
        //  dist_l2 = norm(dsc1.row(i), dsc2.row(j), cv::NORM_L2);
        double dist_l2 = norm(dsc1.row(i), dsc2.row(j), cv::NORM_L2);
        candidates[i][j] = dist_l2;
        points.push_back(kpoints2[j]);

        /*
        // Cross verification with image 1
        cv::Point3f im1pt0(0,-gmlines2[j][2]/gmlines2[j][1],0.);
        cv::Point3f im1pt1(lx,-(gmlines2[j][2]+gmlines2[j][0]*lx)/gmlines2[j][1],0.);
        cv::Vec3f im1line = EquationLine(cv::Point2f(im1pt0.x,im1pt0.y), cv::Point2f(im1pt1.x,im1pt1.y));

        cv::Point2f im1kp(kpoints1[i].x, kpoints1[i].y);

        if (DistancePointLine(im1kp,im1line) <= th || !bCrossVerification) {
            double dist_l2 = 0.;
            if (vkps1.size() == dsc1.rows && vkps2.size() == dsc2.rows)
                dist_l2  = norm(dsc1.row(i),dsc2.row(j),cv::NORM_L2);
            candidates[i][j] = dist_l2;
        }
        */
      }
    }

    /*
    if (bDraw) {
      cv::viz::Viz3d myWindow("Coordinate Frame");

      // Coordinate system
      myWindow.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem(200));

      // Camera frame
      std::vector<cv::Point3f> camFr = {cv::Point3f(0., ly, 0.), cv::Point3f(lx, ly, 0.), cv::Point3f(lx, 0., 0.), cv::Point3f(0., 0., 0.), cv::Point3f(0., ly, 0.)};
      cv::viz::WPolyLine camFrPoly(camFr, cv::viz::Color::gray());
      myWindow.showWidget("camFrPoly", camFrPoly);

      // Epiplane
      cv::Vec3f line = EquationLine(cv::Point2f(pt0.x, pt0.y), cv::Point2f(pt1.x, pt1.y));
      std::vector<cv::Point3f> lineFr = FrustumLine(line, lx, ly);
      lineFr.push_back(co2);
      lineFr.push_back(lineFr[0]);
      cv::viz::WPolyLine epiplane(lineFr, cv::viz::Color::green());
      myWindow.showWidget("epiplane", epiplane);

      // Candidate points projective rays
      for (size_t j = 0; j < candidates[i].size(); j++) {
        if (candidates[i][j] >= 0.) {
          cv::Point3f kp(kpoints2[j].x, kpoints2[j].y, 0.);
          cv::viz::WLine ptLine(co2, kp, cv::viz::Color::red());
          myWindow.showWidget("ptLine" + j, ptLine);
        }
      }
      myWindow.spin();
    }
    */


    if (points.size() > 0 && bDraw) {
      std::string name = "Epiline " + std::to_string(i + 1) + " / " + std::to_string(vkps1.size()) + " - " + std::to_string(points.size()) + " candidates";
      DrawCandidates(im1, im2, line, kpoints1[i], points, name);
    }
  }

  return candidates;
}

std::vector<std::vector<double>> AngMatcher::MatchAngle2D(float th, bool bCrossVerification, 
                                                        bool bDraw) {

  // Look for match candidates
  std::vector<std::vector<double>> candidates = std::vector<std::vector<double>>(vkps1.size(), std::vector<double>(vkps2.size(), -1.));

  for (size_t i = 0; i < vkps1.size(); i++) {
    cv::Point3f pt0(0, -gmlines1[i][2] / gmlines1[i][1], 0.);
    cv::Point3f pt1(lx, -(gmlines1[i][2] + gmlines1[i][0] * lx) / gmlines1[i][1], 0.);
    cv::Vec4f pi = EquationPlane(pt0, pt1, co2);

    for (size_t j = 0; j < kpoints2.size(); j++) {
      cv::Vec3f kp(kpoints2[j].x, kpoints2[j].y, 0.);
      cv::Vec3f v(kp(0) - co2.x, kp(1) - co2.y, kp(2) - co2.z);

      if (AngleLinePlane(pi, v) <= th) {
        // Cross verification with image 1
        cv::Point3f im1pt0(0, -gmlines2[j][2] / gmlines2[j][1], 0.);
        cv::Point3f im1pt1(lx, -(gmlines2[j][2] + gmlines2[j][0] * lx) / gmlines2[j][1], 0.);
        cv::Vec4f im1pi = EquationPlane(im1pt0, im1pt1, co2);

        cv::Vec3f im1kp(kpoints1[i].x, kpoints1[i].y, 0.);
        cv::Vec3f im1v(im1kp(0) - co2.x, im1kp(1) - co2.y, im1kp(2) - co2.z);

        if (AngleLinePlane(im1pi, im1v) <= th || !bCrossVerification)
        {
          //double dist_l2 = 0.;
          //if (vkps1.size() == dsc1.rows && vkps2.size() == dsc2.rows)
          //  dist_l2 = norm(dsc1.row(i), dsc2.row(j), cv::NORM_L2);
          double dist_l2 = norm(dsc1.row(i), dsc2.row(j), cv::NORM_L2);
          candidates[i][j] = dist_l2;
        }
      }
    }

    if (bDraw) {
      cv::viz::Viz3d myWindow("Coordinate Frame");

      // Coordinate system
      myWindow.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem(200));

      // Camera frame
      std::vector<cv::Point3f> camFr = {cv::Point3f(0., ly, 0.), cv::Point3f(lx, ly, 0.), cv::Point3f(lx, 0., 0.), cv::Point3f(0., 0., 0.), cv::Point3f(0., ly, 0.)};
      cv::viz::WPolyLine camFrPoly(camFr, cv::viz::Color::gray());
      myWindow.showWidget("camFrPoly", camFrPoly);

      // Epiplane
      cv::Vec3f line = EquationLine(cv::Point2f(pt0.x, pt0.y), cv::Point2f(pt1.x, pt1.y));
      std::vector<cv::Point3f> lineFr = FrustumLine(line, lx, ly);
      lineFr.push_back(co2);
      lineFr.push_back(lineFr[0]);
      cv::viz::WPolyLine epiplane(lineFr, cv::viz::Color::green());
      myWindow.showWidget("epiplane", epiplane);

      // Candidate points projective rays
      for (size_t j = 0; j < candidates[i].size(); j++) {
        if (candidates[i][j] >= 0.) {
          cv::Point3f kp(kpoints2[j].x, kpoints2[j].y, 0.);
          cv::viz::WLine ptLine(co2, kp, cv::viz::Color::red());
          myWindow.showWidget("ptLine" + j, ptLine);
        }
      }
      myWindow.spin();
    }
  }

  return candidates;
}






std::vector<cv::DMatch> AngMatcher::MatchDescriptors(std::vector<std::vector<double>> candidates, cv::Mat desc1, cv::Mat desc2, double th) {
  std::vector<std::vector<double>> sift_descriptor_simmilarity;

  for (size_t i = 0; i < candidates.size(); i++) {
    std::vector<double> row;
    for (size_t j = 0; j < candidates[i].size(); j++) {
      if (candidates[i][j] >= 0.) {
        double dist_l2 = norm(desc1.row(i), desc2.row(j), cv::NORM_L2);
        row.push_back(dist_l2);
      } else {
        row.push_back(-1.);
      }
    }
    sift_descriptor_simmilarity.push_back(row);
  }

  std::vector<cv::DMatch> nn;

  std::vector<double> flat = flatten(sift_descriptor_simmilarity);
  std::vector<int> index = ordered_indices(flat);

  for (size_t i = 0; i < index.size(); i++) {
    int idx_i, idx_j;
    indices_from_flatten_position(idx_i, idx_j, index[i], sift_descriptor_simmilarity[0].size());
    double dist_ij = sift_descriptor_simmilarity[idx_i][idx_j];
    if (dist_ij > 0.0 && dist_ij < th) {
      //std::cout << "dist_[" << idx_i << "-" << idx_j << "] = " << dist_ij << " of " << th << std::endl;
      nn.push_back(cv::DMatch(idx_i, idx_j, dist_ij));
      for (size_t k = 0; k < sift_descriptor_simmilarity.size(); k++)
        sift_descriptor_simmilarity[k][idx_j] = -1.;
      for (size_t k = 0; k < sift_descriptor_simmilarity[idx_i].size(); k++)
        sift_descriptor_simmilarity[idx_i][k] = -1.;
    }
  }

  //std::cout << nn.size() << " matches" << std::endl;
  return nn;
}



std::vector<cv::DMatch> AngMatcher::NNCandidates(std::vector<std::vector<double>> candidates, 
                                                 double th) {
  std::vector<cv::DMatch> nn;

  for (size_t i = 0; i < candidates.size(); i++) {
    int idx_i = -1;
    double dist_i = 1000000.;

    for (size_t j = 0; j < candidates[i].size(); j++) {
      if (candidates[i][j] >= 0. && candidates[i][j] < dist_i){

        double dist_i_tmp = candidates[i][j];

        int idx_j = -1;
        double dist_j = -1000000.;

        for (size_t k = 0; k < candidates.size(); k++)        {
          if (candidates[k][j] >= 0. && candidates[k][j] < dist_j && candidates[k][j] < dist_i_tmp) {
            idx_j = k;
            dist_j = candidates[k][j];
          }
        }

        if (idx_j < 0) {
          idx_i = j;
          dist_i = dist_i_tmp;
        }
      }
    }

    if (dist_i >= 0. && dist_i <= th) {
      cv::DMatch dm;
      dm.queryIdx = i;
      dm.trainIdx = idx_i;
      dm.imgIdx = 0;
      dm.distance = dist_i;
      nn.push_back(dm);

      for (size_t j = 0; j < candidates.size(); j++) {
        candidates[j][idx_i] = -1.;
      }
    }
  }

  return nn;
}



std::vector<cv::DMatch> AngMatcher::NNCandidates2(std::vector<std::vector<double>> candidates, double th) {
  std::vector<cv::DMatch> nn;

  std::vector<double> flat = flatten(candidates);
  std::vector<int> index = ordered_indices(flat);

  for (size_t i = 0; i < index.size(); i++) {
    int idx_i, idx_j;
    indices_from_flatten_position(idx_i, idx_j, index[i], candidates[0].size());

    if (candidates[idx_i][idx_j] >= 0.) {
      double dist_i = candidates[idx_i][idx_j];
      candidates[idx_i][idx_j] = -1.;

      int idx_j2 = -1;
      double dist_j2 = -1000000.;

      for (size_t j = 0; j < candidates.size(); j++) {
        if (candidates[j][idx_j] >= 0. && candidates[j][idx_j] < dist_j2) {
          dist_j2 = candidates[j][idx_j];
          idx_j2 = j;
        }
      }

      if (idx_j2 == idx_i) {
        cv::DMatch m(idx_i, idx_j, dist_i);
        nn.push_back(m);
      }
    }
  }

  return nn;
}



std::vector<int> AngMatcher::GetPointIndices(const std::vector<cv::DMatch> &matches1, const std::vector<cv::DMatch> &matches2) {
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






void ResizeAndDisplay(const std::string &title, const cv::Mat &img1, float factor, bool wait) {
  cv::Mat out1;
  cv::resize(img1, out1, cv::Size(), factor, factor);

  if (wait) {
    cv::namedWindow(title);
    cv::imshow(title, out1);
    cv::waitKey(0);
    cv::destroyWindow(title);
  } else {
    cv::imshow(title, out1);
  }
}



void ResizeAndDisplay(const std::string &title, const std::vector<cv::Mat> &imgs, float factor, bool wait) {
  int nImgs=imgs.size();
  int imgsInRow=1;
  int imgsInCol=ceil(nImgs/imgsInRow); // You can set this explicitly

  std::vector<cv::Mat> imgs2;
  for(int i=0;i<nImgs;i++) {
      cv::Mat tmp;
      cv::resize(imgs[i], tmp, cv::Size(), factor, factor);
      imgs2.push_back(tmp);
  }

  int cellSizeW = imgs2[0].cols;
  int cellSizeH = imgs2[0].rows;

  int resultImgW = imgs2[0].cols*imgsInRow;
  int resultImgH = imgs2[0].rows*imgsInCol;

  cv::Mat resultImg = cv::Mat::zeros(resultImgH, resultImgW, CV_8UC3);
  unsigned int ind=0;
  cv::Mat tmp;
  for(int i=0;i<imgsInCol;i++) {
    for(int j=0;j<imgsInRow;j++) {
      if(ind<imgs2.size()) {
      int cell_row = i*cellSizeH;
      int cell_col = j*cellSizeW;
      imgs2[ind].copyTo(resultImg(cv::Range(cell_row, cell_row+imgs2[ind].rows),
                                  cv::Range(cell_col, cell_col+imgs2[ind].cols)));
      }
      ind++;
    }
  }
  imshow(title, resultImg);
}



} // namespace am