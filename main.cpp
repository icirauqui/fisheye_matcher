#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//Visualization
#include <opencv2/viz.hpp>

#include <math.h>

#include <iostream>

float rad_to_deg(float rad){
    return rad*180.0/M_PI;
}

float deg_to_rad(float deg){
    return deg*M_PI/180.0;
}

cv::Vec3f equation_line(cv::Point2f p1, cv::Point2f p2){
    float dx = p2.x - p1.x;
    float dy = p2.y - p1.y;
    float m = dy / dx;
    float c = p1.y - m*p1.x;

    cv::Vec3f eqLine(m,-1,c);
    return eqLine;
}

float line_y_x(cv::Vec3f line, float x){
    return (line(0)*x + line(2))/(-line(1));
}

float line_x_y(cv::Vec3f line, float y){
    return (line(1)*y + line(2))/(-line(0));
}

std::vector<cv::Point3f> frustum_line(cv::Vec3f line, float lx, float ly){
    cv::Point3f pt0(0., line_y_x(line,0.), 0.);
    if (pt0.y < 0)
        pt0.y = 0;
    else if (pt0.y > ly)
        pt0.y = ly;
    pt0.x = line_x_y(line,pt0.y);

    cv::Point3f pt1(lx, line_y_x(line,lx), 0.);
    if (pt1.y < 0)
        pt1.y = 0;
    else if (pt1.y > ly)
        pt1.y = ly;
    pt1.x = line_x_y(line,pt1.y);

    std::vector<cv::Point3f> frustum = {pt0,pt1};
    return frustum;
}

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

float angle_line_plane(cv::Vec4f pi, cv::Vec3f v){        
    //Vector normal del plano
    cv::Vec3f n(pi(0), pi(1), pi(2));

    float num = abs(v(0)*n(0) + v(1)*n(1) + v(2)*n(2));
    float den1 = sqrt(v(0)*v(0) + v(1)*v(1) + v(2)*v(2));
    float den2 = sqrt(n(0)*n(0) + n(1)*n(1) + n(2)*n(2));
    
    float beta = acos(num / (den1 * den2));
    float alpha = (M_PI/2) - beta;
    return alpha;
}

void resize_and_display(const std::string& title, const cv::Mat& img1, float factor){
    cv::Mat out1;
    cv::resize(img1, out1, cv::Size(), factor, factor);
    cv::imshow(title,out1);
}
 
static float distancePointLine(const cv::Point2f point, const cv::Vec3f& line){
    return std::fabs(line(0)*point.x + line(1)*point.y + line(2)) / std::sqrt(line(0)*line(0)+line(1)*line(1));
}

static void drawEpipolarLines(const std::string& title, const cv::Mat F,
                const cv::Mat& img1, const cv::Mat& img2,
                const std::vector<cv::Point2f> points1, const std::vector<cv::Point2f> points2,
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
    cv::Scalar color(cv::RNG(256),cv::RNG(256),cv::RNG(256));
 
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

    for (size_t i=0; i<points2.size(); i++){
        cv::circle(outImg(rect2), points2[i], 3, cv::Scalar(50,50,50), -1, cv::LINE_AA);
    }

    for(size_t i=0; i<points1.size(); i++) {
        cv::Scalar color(cv::RNG(256),cv::RNG(256),cv::RNG(256));
    
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
    bool bDraw = false;
    float th_alpha = 0.0174533; //1 deg
    //float th_alpha = 0.0349066; //2 deg
    //float th_alpha = 0.0523599; //3 deg
    //float th_alpha = 0.0698132; //4 deg

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
    float f = (lx/(lx+ly))*fx + (ly/(lx+ly))*fy;
    cv::Point3f c2(cx,cy,f);


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


    // Epipolar matching, compute F and epilines
    cv::Mat F12 = cv::findFundamentalMat(points1,points2);
    //drawEpipolarLines("epip1",F12,im1,im2,points1,points2);

    std::vector<cv::Point2f> kpoints1, kpoints2;
    for (size_t i=0; i<kps1.size(); i++)
        kpoints1.push_back(kps1[i].pt);
    for (size_t i=0; i<kps2.size(); i++)
        kpoints2.push_back(kps2[i].pt);

    std::vector<cv::Vec3f> gmlines1, gmlines2;
    cv::computeCorrespondEpilines(kpoints1, 1, F12, gmlines1);
    cv::computeCorrespondEpilines(kpoints2, 2, F12, gmlines2);
    //drawEpipolarLines("epip2",F12,im1,im2,kpoints1,gmlines1,kpoints2);

    std::vector<std::vector<int> > match_candidates;
    for (size_t i=0; i<kps1.size(); i++){
        cv::Point3f pt0(0,-gmlines1[i][2]/gmlines1[i][1],0.);
        cv::Point3f pt1(im1.cols,-(gmlines1[i][2]+gmlines1[i][0]*im1.cols)/gmlines1[i][1],0.);
        cv::Vec4f pi = equation_plane(pt0, pt1, c2);

        std::vector<int> match_candidates_i;
        for (size_t j=0; j<kpoints2.size(); j++){
            cv::Vec3f kp(kpoints2[j].x, kpoints2[j].y, 0.);
            cv::Vec3f v(kp(0)-c2.x, kp(1)-c2.y, kp(2)-c2.z);

            if (angle_line_plane(pi,v) <= th_alpha){
                match_candidates_i.push_back(j);
            }
        }

        match_candidates.push_back(match_candidates_i);

        if (bDraw){
            cv::viz::Viz3d myWindow("Coordinate Frame");

            // Coordinate system
            myWindow.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem(200));

            // Camera frame
            std::vector<cv::Point3f> camFr = {cv::Point3f(0.,ly,0.), cv::Point3f(lx,ly,0.), cv::Point3f(lx,0.,0.), cv::Point3f(0.,0.,0.), cv::Point3f(0.,ly,0.)};
            cv::viz::WPolyLine camFrPoly(camFr,cv::viz::Color::gray());
            myWindow.showWidget("camFrPoly", camFrPoly);

            // Epiplane
            cv::Vec3f line = equation_line(cv::Point2f(pt0.x,pt0.y), cv::Point2f(pt1.x,pt1.y));
            std::vector<cv::Point3f> lineFr = frustum_line(line,lx,ly);
            lineFr.push_back(c2);
            lineFr.push_back(lineFr[0]);
            cv::viz::WPolyLine epiplane(lineFr,cv::viz::Color::green());
            myWindow.showWidget("epiplane", epiplane);

            // Candidate points projective rays
            for (size_t j=0; j<match_candidates_i.size(); j++){
                cv::Point3f kp(kpoints2[match_candidates_i[j]].x, kpoints2[match_candidates_i[j]].y, 0.);
                cv::viz::WLine ptLine(c2, kp, cv::viz::Color::red());
                myWindow.showWidget("ptLine"+j, ptLine);
            }

            myWindow.spin();
        }
      

    }



    cv::Ptr<cv::DescriptorMatcher> matcher_1 = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<cv::DMatch> > knn_matches_1;

    std::vector<std::vector<double> > distances;
    for (size_t i=0; i<match_candidates.size(); i++){
        std::vector<double> distances_i;
        for (size_t j=0; j<match_candidates[i].size(); j++){
            double dist_l2  = norm(desc1.row(i),desc2.row(match_candidates[i][j]),cv::NORM_L2);
            distances_i.push_back(dist_l2);
        }
        distances.push_back(distances_i);
    }

    // For each kp2 to which kp1s has been marked as candidate
    std::vector<std::vector<int> > inv_index;
    std::vector<std::vector<double> > inv_distances;
    for (size_t i=0; i<kps2.size(); i++){
        std::vector<int> inv_index_i;
        std::vector<double> inv_distances_i;
        for (size_t j=0; j<match_candidates.size(); j++){
            for (size_t k=0; k<match_candidates[j].size(); k++){
                if (match_candidates[j][k]==i){
                    inv_index_i.push_back(j);
                    inv_distances_i.push_back(distances[j][k]);
                }
            }
        }
        inv_index.push_back(inv_index_i);
        inv_distances.push_back(inv_distances_i);
    }


    std::vector<cv::DMatch> egmatches;
    for (size_t i=0; i<inv_distances.size(); i++){
        double score = -1.0;
        int id_1 = -1;
        int id_2 = -1;
        if (inv_distances[i].size() > 0){
            for (size_t j=0; j<inv_distances[i].size(); j++){
                if (inv_distances[i][j] > score && inv_distances[i][j] > -1.0){
                    score = inv_distances[i][j];
                    id_1 = inv_index[i][j];
                    id_2 = j;
                }
            }
        }
        if (id_1 > -1){
            for (size_t j=0; j<inv_index.size(); j++){
                for (size_t k=0; k<inv_index[j].size(); k++){
                    if (inv_index[j][k]==id_1){
                        inv_index[j][k] = -1;
                        inv_distances[j][k] = -1.0;
                    }
                }
            }

            cv::DMatch dm;
            dm.queryIdx = id_1;
            dm.trainIdx = id_2;
            dm.imgIdx = 0;
            for (size_t j=0; j<match_candidates[id_1].size(); j++){
                if (match_candidates[id_1][j] == id_2){
                    dm.distance = (float) distances[id_1][j];
                    break;
                }
            }
            egmatches.push_back(dm);
        }
    }


    std::vector<cv::DMatch> egmatches1;
    double th = 1000.0;
    for (size_t i=0; i<egmatches.size(); i++){
        if (egmatches[i].distance <= th){
            egmatches1.push_back(egmatches[i]);
        }
    }

    std::cout << "good / matches / kps2 = " << egmatches1.size() << " / " << egmatches.size() << " / " << kps2.size() << std::endl;

    cv::Mat imout;
    cv::drawMatches(im1,kps1,im2,kps2,egmatches1,imout);
    resize_and_display("eg",imout,0.5);

    cv::waitKey(0);
    return 0;
}