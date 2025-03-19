#include <iostream>
#include <opencv2/opencv.hpp>
#include <limits>


// Replace these values with the values obtained from your camera calibration
double k1 = -0.139659;
double k2 = -0.000313999;
double k3 = 0.0015055;
double k4 = -0.000131671;
std::vector<double> coefficients = {k1, k2, k3, k4};
double fx = 717.691;
double fy = 718.021;
double cx = 734.728;
double cy = 552.072;

double max_double = std::numeric_limits<double>::max();
double min_double = std::numeric_limits<double>::min();



cv::Mat initUndistortRectifyMap(cv::Size size) {
    cv::Mat map1(size, CV_32FC1);
    cv::Mat map2(size, CV_32FC1);

    float s = 0;  // Skew is assumed to be 0

    float fx_inv = 1.0 / fx;
    float fy_inv = 1.0 / fy;

    for (int y = 0; y < size.height; ++y) {
        for (int x = 0; x < size.width; ++x) {
            float x_normalized = (x - cx) * fx_inv;
            float y_normalized = (y - cy) * fy_inv;

            float r_sq = x_normalized * x_normalized + y_normalized * y_normalized;
            float r = std::sqrt(r_sq);

            float theta = std::atan(r);

            float theta_distorted;
            if (r == 0) {
                theta_distorted = 1.0;
            } else {
                theta_distorted = theta * (1.0 + k1 * r_sq + k2 * r_sq * r_sq + k3 * r_sq * r_sq * r_sq + k4 * r_sq * r_sq * r_sq * r_sq);
            }

            float x_distorted = theta_distorted * x_normalized / r;
            float y_distorted = theta_distorted * y_normalized / r;

            float x_distorted_pixel = x_distorted * fx + cx;
            float y_distorted_pixel = y_distorted * fy + cy;

            map1.at<float>(y, x) = x_distorted_pixel;
            map2.at<float>(y, x) = y_distorted_pixel;
        }
    }

    cv::Mat map;
    cv::merge(std::vector<cv::Mat>{map1, map2}, map);

    return map;
}


cv::Mat initDistortRectifyMap(cv::Size size) {
    cv::Mat map1(size, CV_32FC1);
    cv::Mat map2(size, CV_32FC1);

    float s = 0;  // Skew is assumed to be 0

    float fx_inv = 1.0 / fx;
    float fy_inv = 1.0 / fy;

    cv::Mat R = cv::Mat::eye(3, 3, CV_64FC1);

    for (int y = 0; y < size.height; ++y) {
        for (int x = 0; x < size.width; ++x) {
            cv::Mat point_in(3, 1, CV_64FC1);
            point_in.at<double>(0, 0) = (x - cx) * fx_inv;
            point_in.at<double>(1, 0) = (y - cy) * fy_inv;
            point_in.at<double>(2, 0) = 1;

            cv::Mat point_out = R * point_in;

            float x_normalized = point_out.at<double>(0, 0) / point_out.at<double>(2, 0);
            float y_normalized = point_out.at<double>(1, 0) / point_out.at<double>(2, 0);

            float r_sq = x_normalized * x_normalized + y_normalized * y_normalized;
            float r = std::sqrt(r_sq);

            float theta = std::atan(r);

            float theta_distorted = theta * (1.0 + k1 * r_sq + k2 * r_sq * r_sq + k3 * r_sq * r_sq * r_sq + k4 * r_sq * r_sq * r_sq * r_sq);

            float x_distorted = theta_distorted * x_normalized / r;
            float y_distorted = theta_distorted * y_normalized / r;

            float x_distorted_pixel = x_distorted * fx + cx;
            float y_distorted_pixel = y_distorted * fy + cy;

            map1.at<float>(y, x) = x_distorted_pixel;
            map2.at<float>(y, x) = y_distorted_pixel;
        }
    }

    cv::Mat map;
    cv::merge(std::vector<cv::Mat>{map1, map2}, map);

    return map;
}



void undistortKB(const cv::Mat& src, cv::Mat& dst) {
    int width = src.cols;
    int height = src.rows;

    dst = cv::Mat(height, width, src.type());

    for (int v = 0; v < height; v++) {
        for (int u = 0; u < width; u++) {
            // Normalize image coordinates
            double x_d = (u - cx) / fx;
            double y_d = (v - cy) / fy;

            // Compute distorted radius
            double r_d = std::sqrt(x_d * x_d + y_d * y_d);
            double th_d = std::atan(r_d);

            // Compute undistorted radius
            //double r_u = k1 * r_d + k2 * std::pow(r_d, 3) + k3 * std::pow(r_d, 5) + k4 * std::pow(r_d, 7);
            double th_u = th_d * (1.0 + k1*std::pow(th_d, 2) + k2*std::pow(th_d, 4) + k3*std::pow(th_d, 6) + k4*std::pow(th_d, 8));

            double r_u = th_u / th_d * r_d;

            // Compute undistorted normalized coordinates
            double x_u = x_d * (r_u / r_d);
            double y_u = y_d * (r_u / r_d);

            // Compute undistorted pixel coordinates
            double u_undistorted = x_u * fx + cx;
            double v_undistorted = y_u * fy + cy;

            // Perform bilinear interpolation
            if (u_undistorted >= 0 && u_undistorted < width - 1 && v_undistorted >= 0 && v_undistorted < height - 1) {
                int u1 = static_cast<int>(u_undistorted);
                int v1 = static_cast<int>(v_undistorted);
                int u2 = u1 + 1;
                int v2 = v1 + 1;

                double x_weight = u_undistorted - u1;
                double y_weight = v_undistorted - v1;

                cv::Vec3b p1 = src.at<cv::Vec3b>(v1, u1);
                cv::Vec3b p2 = src.at<cv::Vec3b>(v1, u2);
                cv::Vec3b p3 = src.at<cv::Vec3b>(v2, u1);
                cv::Vec3b p4 = src.at<cv::Vec3b>(v2, u2);

                cv::Vec3b pixel_value = (1 - x_weight) * (1 - y_weight) * p1 +
                                        x_weight * (1 - y_weight) * p2 +
                                        (1 - x_weight) * y_weight * p3 +
                                        x_weight * y_weight * p4;

                dst.at<cv::Vec3b>(v, u) = pixel_value;
            }
        }
    }
}



void undistortKBu(const cv::Mat& src, cv::Mat& dst) {
    int width = src.cols;
    int height = src.rows;

    dst = cv::Mat(height, width, src.type());

    double u_min = max_double;
    double u_max = min_double;
    double v_min = max_double;
    double v_max = min_double;
    
    for (int v = 0; v < height; v++) {
        for (int u = 0; u < width; u++) {
            // Normalize image coordinates
            double x_d = (u - cx) / fx;
            double y_d = (v - cy) / fy;

            // Compute distorted radius
            double r_d = std::sqrt(x_d * x_d + y_d * y_d);
            double th_d = std::atan(r_d);

            // Compute undistorted radius
            //double r_u = k1 * r_d + k2 * std::pow(r_d, 3) + k3 * std::pow(r_d, 5) + k4 * std::pow(r_d, 7);
            double th_u = th_d * (1.0 + k1*std::pow(th_d, 2) + k2*std::pow(th_d, 4) + k3*std::pow(th_d, 6) + k4*std::pow(th_d, 8));

            double r_u = th_u / th_d * r_d;

            // Compute undistorted normalized coordinates
            double x_u = x_d * (r_u / r_d);
            double y_u = y_d * (r_u / r_d);

            // Compute undistorted pixel coordinates
            double u_undistorted = x_u * fx + cx;
            double v_undistorted = y_u * fy + cy;

            //std::cout << u << " " << v << "    " << u_undistorted << "\t" << v_undistorted << std::endl;

            if (u_undistorted < u_min) u_min = u_undistorted;
            if (u_undistorted > u_max) u_max = u_undistorted;
            if (v_undistorted < v_min) v_min = v_undistorted;
            if (v_undistorted > v_max) v_max = v_undistorted;

            // Perform bilinear interpolation
            if (u_undistorted >= 0 && u_undistorted < width - 1 && v_undistorted >= 0 && v_undistorted < height - 1) {
                int u1 = static_cast<int>(u_undistorted);
                int v1 = static_cast<int>(v_undistorted);
                int u2 = u1 + 1;
                int v2 = v1 + 1;

                double x_weight = u_undistorted - u1;
                double y_weight = v_undistorted - v1;

                cv::Vec3b p1 = src.at<cv::Vec3b>(v1, u1);
                cv::Vec3b p2 = src.at<cv::Vec3b>(v1, u2);
                cv::Vec3b p3 = src.at<cv::Vec3b>(v2, u1);
                cv::Vec3b p4 = src.at<cv::Vec3b>(v2, u2);

                cv::Vec3b pixel_value = (1 - x_weight) * (1 - y_weight) * p1 +
                                        x_weight * (1 - y_weight) * p2 +
                                        (1 - x_weight) * y_weight * p3 +
                                        x_weight * y_weight * p4;

                dst.at<cv::Vec3b>(v, u) = pixel_value;
            } else {
                std::cout << "out of image bounds" << std::endl;
            }


        }
    }

    std::cout << "src size: " << src.size() << std::endl;
    //dst = cv::Mat(v_max - v_min + 1, u_max - u_min + 1, src.type());
    std::cout << "dst size: " << dst.size() << std::endl;
    std::cout << "dst size: [" << v_max - v_min + 1 << " x " << u_max - u_min + 1 << "]" << std::endl;

}





int main() {

    std::string image_i = "/home/icirauqui/workspace_phd/fisheye_matcher/images/s1_001.png";
    std::string image_o = "/home/icirauqui/workspace_phd/fisheye_matcher/images/s1_001_o.png";

    cv::Mat src = cv::imread(image_i);
    if (src.empty()) {
        std::cerr << "Failed to load the image." << std::endl;
        return -1;
    }


    //cv::Mat dst;
    //undistortKBu(src, dst);

    // New black image of size src
    cv::Mat blank = cv::Mat::zeros(src.size(), src.type());
    // Draw line in image
    cv::line(blank, cv::Point(0, 1000), cv::Point(1000, 0), cv::Scalar(0, 255, 255), 4, 8);


    // Compute the undistortion map
    cv::Mat map_und = initDistortRectifyMap(src.size());
    cv::Mat map_dis = initUndistortRectifyMap(src.size());

    // Apply the fisheye distortion
    cv::Mat dst, dst2, blank2;
    cv::remap(src, dst, map_und, cv::Mat(), cv::INTER_LINEAR, cv::BORDER_WRAP);
    cv::remap(blank, blank, map_und, cv::Mat(), cv::INTER_LINEAR, cv::BORDER_WRAP);
    cv::remap(dst, dst2, map_dis, cv::Mat(), cv::INTER_LINEAR, cv::BORDER_WRAP);
    cv::remap(blank, blank2, map_dis, cv::Mat(), cv::INTER_LINEAR, cv::BORDER_WRAP);


    // Resize all images
    cv::resize(src, src, cv::Size(), 0.5, 0.5);
    cv::resize(dst, dst, cv::Size(), 0.5, 0.5);
    cv::resize(blank, blank, cv::Size(), 0.5, 0.5);
    cv::resize(dst2, dst2, cv::Size(), 0.5, 0.5);
    cv::resize(blank2, blank2, cv::Size(), 0.5, 0.5);


    // Show image
    cv::imshow("src", src);
    cv::imshow("dst", dst);
    cv::imshow("blank", blank);
    cv::imshow("dst2", dst2);
    cv::imshow("blank2", blank2);
    cv::waitKey(0);

    //cv::Mat undistortedImage = undistortImageKB2(src);


    //cv::imwrite(image_o, undistortedImage);

    return 0;
}