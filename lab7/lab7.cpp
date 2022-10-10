#include <opencv2/video/background_segm.hpp>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/video/tracking.hpp>
#include <opencv2/core/ocl.hpp>

#define MINTRACKAREA 50

using namespace std;
using namespace cv;

void detectAndDisplay(int, void*);
int displayLukas(const string& filename);
int tracking(const string& filename);

static const int MAX_CORNERS = 1000;

int main(int argc, const char** argv)
{

    detectAndDisplay(0, 0);
    displayLukas("D:/univer/PMMP/image/cars.mp4");
    tracking("D:/univer/PMMP/image/cars.mp4");
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

int displayLukas(const string& filename) {
    // Read the video 
    VideoCapture capture(filename);
    if (!capture.isOpened()) {
        //error in opening the video input
        cerr << "Unable to open file!" << endl;
        return 0;
    }

    // Create random colors
    vector<Scalar> colors;
    RNG rng;
    for (int i = 0; i < 100; i++)
    {
        int r = rng.uniform(0, 256);
        int g = rng.uniform(0, 256);
        int b = rng.uniform(0, 256);
        colors.push_back(Scalar(r, g, b));
    }

    Mat old_frame, old_gray;
    vector<Point2f> p0, p1;

    // Read first frame and find corners in it
    capture >> old_frame;
    cvtColor(old_frame, old_gray, COLOR_BGR2GRAY);
    goodFeaturesToTrack(old_gray, p0, 100, 0.3, 7, Mat(), 7, false, 0.04);

    // Create a mask image for drawing purposes
    Mat mask = Mat::zeros(old_frame.size(), old_frame.type());

    while (true) {
        int counter = 0;
        // Read new frame
        Mat frame, frame_gray;
        capture >> frame;
        if (frame.empty())
            break;
        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

        // Calculate optical flow
        vector<uchar> status;
        vector<float> err;
        TermCriteria criteria = TermCriteria((TermCriteria::COUNT)+(TermCriteria::EPS), 10, 0.03);
        calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, Size(15, 15), 2, criteria);
        vector<Point2f> good_new;

        // Visualization part
        for (uint i = 0; i < p0.size(); i++)
        {
            // Select good points
            if (status[i] == 1) {
                good_new.push_back(p1[i]);
                // Draw the tracks
                line(mask, p1[i], p0[i], colors[i], 2);
                circle(frame, p1[i], 5, colors[i], -1);
            }
        }

        // Display the demo
        Mat img;
        add(frame, mask, img);
        //if (save) {
            string save_path = "./optical_flow_frames/frame_" + to_string(counter) + ".jpg";
            imwrite(save_path, img);
        //}
        imshow("flow", img);
        int keyboard = waitKey(25);
        if (keyboard == 'q' || keyboard == 27)
            break;

        // Update the previous frame and previous points
        old_gray = frame_gray.clone();
        p0 = good_new;
        counter++;
    }
}

int tracking(const string& filename) {
    // List of tracker types in OpenCV 3.4.1
    string trackerTypes[8] = { "BOOSTING", "MIL", "KCF", "TLD","MEDIANFLOW", "GOTURN", "MOSSE", "CSRT" };
    // vector <string> trackerTypes(types, std::end(types));

    // Create a tracker
    string trackerType = trackerTypes[1];

    Ptr<Tracker> tracker = TrackerMIL::create();;

    */
    // Read video
    VideoCapture video(filename);

    // Exit if video is not opened
    if (!video.isOpened())
    {
        cout << "Could not read video file" << endl;
        return 1;
    }

    // Read first frame 
    Mat frame;
    bool ok = video.read(frame);

    // Uncomment the line below to select a different bounding box 
    Rect bbox = selectROI(frame, false); 
    // Display bounding box. 
    rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);

    imshow("Tracking", frame);
    tracker->init(frame, bbox);

    while (video.read(frame))
    {
        // Start timer
        double timer = (double)getTickCount();

        // Update the tracking result
        bool ok = tracker->update(frame, bbox);

        // Calculate Frames per second (FPS)
        float fps = getTickFrequency() / ((double)getTickCount() - timer);

        if (ok)
        {
            // Tracking success : Draw the tracked object
            rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);
        }
        else
        {
            // Tracking failure detected.
            putText(frame, "Tracking failure detected", Point(100, 80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
        }

        // Display tracker type on frame
        putText(frame, trackerType + " Tracker", Point(100, 20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);

        // Display frame.
        imshow("Tracking", frame);

        // Exit if ESC pressed.
        int k = waitKey(1);
        if (k == 27)
        {
            break;
        }

    }
}