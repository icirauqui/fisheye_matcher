#include <opencv2/viz.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
using namespace cv;
using namespace std;




int main()
{
    viz::Viz3d myWindow("Coordinate Frame");
    myWindow.showWidget("Coordinate Widget", viz::WCoordinateSystem());
myWindow.spin();

    return 0;
}