#include <iostream>


#include <opencv2/highgui/highgui.hpp>

int main() {
    std::cout << "Hello World!";

  cv::Mat im1 = cv::imread("images/1.png", cv::IMREAD_COLOR);

  // Show image 
  cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
  cv::imshow("Display window", im1);
  cv::waitKey(0);


    return 0;
}