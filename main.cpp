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
    std::cout << "eqLine = " << eqLine << std::endl;
    return eqLine;
}


float line_y_x(cv::Vec3f line, float x){
    return (line(0)*x + line(2))/(-line(1));
}

float line_x_y(cv::Vec3f line, float y){
    return (line(1)*y + line(2))/(-line(0));
}

std::vector<cv::Point3f> frustum_line(cv::Vec3f line, float lx, float ly){
    float x0, y0, z0, x1, y1, z1;

    x0 = 0;
    y0 = line_y_x(line,x0);
    z0 = 0;
    if (y0 < 0){
        y0 = 0;
        x0 = line_x_y(line,y0);
    }
    else if (y0 > ly){
        y0 = ly;
        x0 = line_x_y(line,y0);
    }
    cv::Point3f pt0(x0,y0,z0);

    x1 = lx;
    y1 = line_y_x(line,x1);
    z1 = 0;
    std::cout << x1 << " " << y1 << " " << z1 << std::endl;
    if (y1 < 0){
        y1 =ly;
        x1 = line_x_y(line,y1);
    }
    else if (y1 > ly){
        y1 = ly;
        x1 = line_x_y(line,y1);
    }
    cv::Point3f pt1(x1,y1,z1);

    std::vector<cv::Point3f> frustum = {pt0,pt1};
    
    return frustum;
    
}

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

    //std::cout << "Equation plane = " << a << " x " << b << " y " << c << " z " << d << std::endl;

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
 
static float distancePointLine(const cv::Point2f point, const cv::Vec3f& line)
{
    //Line is given as a*x + b*y + c = 0
    return std::fabs(line(0)*point.x + line(1)*point.y + line(2)) / std::sqrt(line(0)*line(0)+line(1)*line(1));
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

    bool bDraw = false;
    float th_alpha = 0.0698132; //4 deg

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

    //cv::computeCorrespondEpilines(points1, 1, F12, gmlines1);
    //cv::computeCorrespondEpilines(points2, 2, F12, gmlines2);
    //drawEpipolarLines("epip2",F12,im1,im2,points1,gmlines1,points2);

    cv::Vec3f c2v(cx,cy,f);
    cv::Point3f c2(cx,cy,f);


    cv::viz::Viz3d myWindow("Coordinate Frame");
    if (bDraw) {
        // Coordinate system
        myWindow.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem(200));

        // Camera frame
        cv::Point3f fr00(0.,ly,0.);
        cv::Point3f fr01(lx,ly,0.);
        cv::Point3f fr10(0.,0.,0.);
        cv::Point3f fr11(lx,0.,0.);
        cv::viz::WLine wfr00(fr00,fr01,cv::viz::Color::gray());
        cv::viz::WLine wfr01(fr01,fr11,cv::viz::Color::gray());
        cv::viz::WLine wfr10(fr11,fr10,cv::viz::Color::gray());
        cv::viz::WLine wfr11(fr10,fr00,cv::viz::Color::gray());
        myWindow.showWidget("frame00", wfr00);
        myWindow.showWidget("frame01", wfr01);
        myWindow.showWidget("frame10", wfr10);
        myWindow.showWidget("frame11", wfr11);
    }

    std::vector<cv::Point2f> gmpoints1, gmpoints2;
    std::vector< std::vector<cv::DMatch> > gm_matches;
    std::vector<std::vector<int> > match_candidates;

    for (size_t i=0; i<kps1.size(); i++){
        cv::Point3f pt0(0,-gmlines1[i][2]/gmlines1[i][1],0.);
        cv::Point3f pt1(im1.cols,-(gmlines1[i][2]+gmlines1[i][0]*im1.cols)/gmlines1[i][1],0.);
        cv::Vec4f pi = equation_plane(pt0, pt1, c2);

        if (bDraw){
            cv::Vec3f line = equation_line(cv::Point2f(pt0.x,pt0.y), cv::Point2f(pt1.x,pt1.y));
            std::vector<cv::Point3f> pts3d2 = frustum_line(line,lx,ly);
            cv::viz::WCloud cloud_widget2(pts3d2, cv::viz::Color::green());
            cv::viz::WLine epiline(pts3d2[0],pts3d2[1],cv::viz::Color::green());
            myWindow.showWidget("epiLine", epiline);

            pts3d2.push_back(c2);
            pts3d2.push_back(pts3d2[0]);
            cv::viz::WPolyLine vizPi(pts3d2,cv::viz::Color::green());
            myWindow.showWidget("vizPi", vizPi);

            myWindow.spinOnce();
        }

        std::vector<int> match_candidates_i;

        for (size_t j=0; j<kpoints2.size(); j++){
        //for (size_t j=0; j<points2.size(); j++){
            cv::Vec3f kp(kpoints2[j].x, kpoints2[j].y, 0.);
            //cv::Vec3f kp(points2[j].x, points2[j].y, 0.);

            // Director vector of line
            cv::Vec3f v(kp(0)-c2v(0), kp(1)-c2v(1), kp(2)-c2v(2));
            
            float alpha = angle_line_plane(pi,v);


            if (alpha <= th_alpha){
                match_candidates_i.push_back(j);
            }

            if (bDraw) {
                std::cout << "Angle " << alpha << " rad, " << rad_to_deg(alpha) << " deg" << std::endl;
                cv::viz::WLine ptLine(c2, cv::Point3f(kp(0),kp(1),kp(2)), cv::viz::Color::red());
                myWindow.showWidget("ptLine"+j, ptLine);
                myWindow.spinOnce();
            }
        }

        match_candidates.push_back(match_candidates_i);

        if (bDraw){
            myWindow.spin();
        }
      

    }



    cv::Ptr<cv::DescriptorMatcher> matcher_1 = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<cv::DMatch> > knn_matches_1;

    std::cout << desc1.size() << "  " << match_candidates.size() << std::endl;
    std::cout << desc1.rows << "  " << desc1.cols << std::endl;
    //matcher_1->knnMatch( desc1, desc2, knn_matches_1, 2, match_candidates);

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

    std::cout << "kps2 = " << kps2.size() << std::endl;
    std::cout << "matches = " << egmatches.size() << std::endl;

    std::vector<cv::DMatch> egmatches1;
    double th = 400.0;
    for (size_t i=0; i<egmatches.size(); i++){
        if (egmatches[i].distance <= th){
            egmatches1.push_back(egmatches[i]);
        }
    }

    std::cout << "matches1 = " << egmatches1.size() << std::endl;


    cv::Mat imout;
    cv::drawMatches(im1,kps1,im2,kps2,egmatches1,imout);
    resize_and_display("eg",imout,0.5);

/*
    for (size_t i=0; i<inv_index.size(); i++){
        if (inv_index[i].size() > 0){
            std::cout << i << "( ";
            for (size_t j=0; j<inv_index[i].size(); j++){
                std::cout << inv_index[i][j] << " ";
            }
            std::cout << " )" << std::endl;
        }
    }
*/

    cv::waitKey(0);
    return 0;
}