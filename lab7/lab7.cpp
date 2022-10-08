#include <opencv2/video/background_segm.hpp>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
using namespace std;
using namespace cv;

void detectAndDisplay(int, void*);

int main(int argc, const char** argv)
{

    detectAndDisplay(0, 0);
}

void detectAndDisplay(int, void*) {
    // Init background substractor
    Ptr<BackgroundSubtractor> bg_model = createBackgroundSubtractorMOG2(500, 16.0, true);
    // Create empy input img, foreground and background image and foreground mask.
    Mat img, foregroundMask, backgroundImage, foregroundImg;
    // capture video from source 0, which is web camera, If you want capture video 
    //file just replace //by  VideoCapture cap("videoFile.mov")
    VideoCapture cap(0);
    // main loop to grab sequence of input files
    for (;;) {
        bool ok = cap.grab();
        if (ok == false) {
            std::cout << "Video Capture Fail" << std::endl;
        }
        else {
            // obtain input image from source
            cap.retrieve(img, CAP_OPENNI_BGR_IMAGE);
            // Just resize input image if you want
            resize(img, img, Size(640, 480));
            // create foreground mask of proper size
            if (foregroundMask.empty()) {
                foregroundMask.create(img.size(), img.type());
            }
            // compute foreground mask 8 bit image
            // -1 is parameter that chose automatically your learning rate
            bg_model->apply(img, foregroundMask, true ? -1 : 0);
            // smooth the mask to reduce noise in image
            GaussianBlur(foregroundMask, foregroundMask, Size(11, 11), 3.5, 3.5);
            // threshold mask to saturate at black and white values
            threshold(foregroundMask, foregroundMask, 10, 255, THRESH_BINARY);
            // create black foreground image
            foregroundImg = Scalar::all(0);
            // Copy source image to foreground image only in area with white mask
            img.copyTo(foregroundImg, foregroundMask);
            //Get background image
            bg_model->getBackgroundImage(backgroundImage);
            // Show the results
            imshow("foreground mask", foregroundMask);
            
            if (!backgroundImage.empty()) {
                imshow("mean background image", backgroundImage);
                if (waitKey(10) == 27)
                {
                    break; // escape
                }
            }
        }
    }
}