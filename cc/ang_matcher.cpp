#include "ang_matcher.h"


namespace am {




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
  cv::waitKey(1);
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
  cv::waitKey(1);
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

/*
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
*/






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





std::vector<std::vector<double>> MatchSampson(std::vector<cv::KeyPoint> vkps1, std::vector<cv::KeyPoint> vkps2,
                                               cv::Mat dsc1, cv::Mat dsc2,
                                               cv::Mat F,
                                               float lx, float ly, cv::Point3f co2,
                                               float th, bool bCrossVerification, bool bDraw) {
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

  // Look for match candidates
  std::vector<std::vector<double>> candidates = std::vector<std::vector<double>>(vkps1.size(), std::vector<double>(vkps2.size(), -1.));

  for (size_t i = 0; i < vkps1.size(); i++) {
    cv::Point3f pt0(0, -gmlines1[i][2] / gmlines1[i][1], 0.);
    cv::Point3f pt1(lx, -(gmlines1[i][2] + gmlines1[i][0] * lx) / gmlines1[i][1], 0.);
    cv::Vec3f line = EquationLine(cv::Point2f(pt0.x, pt0.y), cv::Point2f(pt1.x, pt1.y));

    for (size_t j = 0; j < kpoints2.size(); j++) {
      cv::Point2f kp(kpoints2[j].x, kpoints2[j].y);

      // if (DistancePointLine(kp,line) <= th) {
      if (DistanceSampson(vkps1[i].pt, kp, F)) {
        double dist_l2 = 0.;
        if (vkps1.size() == dsc1.rows && vkps2.size() == dsc2.rows)
          dist_l2 = norm(dsc1.row(i), dsc2.row(j), cv::NORM_L2);
        candidates[i][j] = dist_l2;

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

std::vector<std::vector<double>> MatchAngle(std::vector<cv::KeyPoint> vkps1, std::vector<cv::KeyPoint> vkps2,
                                             cv::Mat dsc1, cv::Mat dsc2,
                                             cv::Mat F,
                                             float lx, float ly, cv::Point3f co2,
                                             float th, bool bCrossVerification, bool bDraw) {
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
          double dist_l2 = 0.;
          if (vkps1.size() == dsc1.rows && vkps2.size() == dsc2.rows)
            dist_l2 = norm(dsc1.row(i), dsc2.row(j), cv::NORM_L2);
          candidates[i][j] = dist_l2;
        }

        // double dist_l2 = 0.;
        // if (vkps1.size() == dsc1.rows && vkps2.size() == dsc2.rows)
        //     dist_l2  = norm(dsc1.row(i),dsc2.row(j),cv::NORM_L2);
        // candidates[i][j] = dist_l2;
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




std::vector<cv::DMatch> NNCandidates(std::vector<std::vector<double>> candidates, double th) {
  std::vector<cv::DMatch> nn;

  for (size_t i = 0; i < candidates.size(); i++)
  {
    int idx_i = -1;
    double dist_i = 1000000.;

    for (size_t j = 0; j < candidates[i].size(); j++)
    {
      if (candidates[i][j] >= 0. && candidates[i][j] < dist_i)
      {

        double dist_i_tmp = candidates[i][j];

        int idx_j = -1;
        double dist_j = -1000000.;

        for (size_t k = 0; k < candidates.size(); k++)
        {
          if (candidates[k][j] >= 0. && candidates[k][j] < dist_j && candidates[k][j] < dist_i_tmp)
          {
            idx_j = k;
            dist_j = candidates[k][j];
          }
        }

        if (idx_j < 0)
        {
          idx_i = j;
          dist_i = dist_i_tmp;
        }
      }
    }

    if (dist_i >= 0. && dist_i <= th)
    {
      cv::DMatch dm;
      dm.queryIdx = i;
      dm.trainIdx = idx_i;
      dm.imgIdx = 0;
      dm.distance = dist_i;
      nn.push_back(dm);

      for (size_t j = 0; j < candidates.size(); j++)
      {
        candidates[j][idx_i] = -1.;
      }
    }
  }

  return nn;
}




void HistogramDMatch(const std::string &title, std::vector<cv::DMatch> matches, int th, int factor) {
  std::vector<int> hist = std::vector<int>(th / factor, 0);
  for (size_t i = 0; i < matches.size(); i++)
  {
    int val = floor(matches[i].distance / factor);
    hist[val]++;
  }

  std::cout << title << " = " << matches.size() << "\t - ";
  for (size_t i = 0; i < hist.size(); i++)
  {
    std::cout << ((i * factor) + factor) << "(" << hist[i] << ") ";
  }
  std::cout << std::endl;
}







void ResizeAndDisplay(const std::string &title, const cv::Mat &img1, float factor) {
  cv::Mat out1;
  cv::resize(img1, out1, cv::Size(), factor, factor);
  cv::imshow(title, out1);
}



void ResizeAndDisplay(const std::string &title, const std::vector<cv::Mat> &imgs, float factor) {
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
  int ind=0;
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