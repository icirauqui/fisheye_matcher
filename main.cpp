#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.h>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;


int main()
{
    Mat frame;

    int imageId = 0;
    String imageName;
    char key;

    int cameraId = 0;
    int camx, camy;
    cout << endl;
    cout << "      Camera Id = "; cin >> cameraId;
    cout << "   Camera Width = "; cin >> camx;
    cout << "  Camera Height = "; cin >> camy;
    cout << endl;

    namedWindow("Camera",CV_WINDOW_AUTOSIZE);
    namedWindow("Captured",CV_WINDOW_AUTOSIZE);

    for (;;){
        VideoCapture cap(cameraId);
        if(cap.isOpened()) {
            cap.set(CV_CAP_PROP_FRAME_WIDTH,camx);
            cap.set(CV_CAP_PROP_FRAME_HEIGHT,camy);
            

            for(;;)
            {
                cap >> frame;
                imshow("Camera",frame);


                key = waitKey(30);
                if (key=='c') {
                    imageId++;
                    imshow("Captured",frame);

                    stringstream itos;
                    itos << imageId;
                    string Id = itos.str();

                    imageName = "./Captures/" + Id + ".png";
                    imwrite(imageName,frame);
                }
            }
        }
        else {
            cout << endl;
            cout << "  Error in camera id" << endl;
            cout << "      Camera Id = "; cin >> cameraId;
            cout << endl;
        }

    }
    
    return 0;
}
