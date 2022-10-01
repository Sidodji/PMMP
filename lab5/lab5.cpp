#include <opencv2\opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat src, src2, src_gray, src_gray2;
int thresh = 200;
int max_thresh = 255;

RNG rng(12345);

int maxCorners = 23;
int maxTrackbar = 100;

void cornerHarris_demo(int, void*);
void feature_detection(int, void*);

int main(int argc, char* argv[])
{
    src = imread("D:/univer/PMMP/image/HarrisImage.jpg", IMREAD_COLOR);
    src2 = imread("D:/univer/PMMP/image/zdanie.jpg", IMREAD_COLOR);

    cvtColor(src, src_gray, COLOR_BGR2GRAY);
    cvtColor(src2, src_gray2, COLOR_BGR2GRAY);

    imshow("Source picture", src);

    cornerHarris_demo(0, 0);

    feature_detection(0, 0);

    waitKey();
    return 0;
}

void cornerHarris_demo(int, void*)
{
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;
    Mat dst = Mat::zeros(src.size(), CV_32FC1);
    cornerHarris(src_gray, dst, blockSize, apertureSize, k);
    Mat dst_norm, dst_norm_scaled;
    normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    convertScaleAbs(dst_norm, dst_norm_scaled);
    for (int i = 0; i < dst_norm.rows; i++)
    {
        for (int j = 0; j < dst_norm.cols; j++)
        {
            if ((int)dst_norm.at<float>(i, j) > thresh)
            {
                circle(dst_norm_scaled, Point(j, i), 5, Scalar(0), 2, 8, 0);
            }
        }
    }
    
    imshow("task 1", dst_norm_scaled);
}

void feature_detection(int, void*) {
    if (maxCorners < 1) { maxCorners = 1; }

    /// Parameters for Shi-Tomasi algorithm
    vector<Point2f> corners;
    double qualityLevel = 0.01;
    double minDistance = 10;
    int blockSize = 3;
    bool useHarrisDetector = false;
    double k = 0.04;

    /// Copy the source image
    Mat copy;
    copy = src2.clone();

    /// Apply corner detection
    goodFeaturesToTrack(src_gray2,
        corners,
        maxCorners,
        qualityLevel,
        minDistance,
        Mat(),
        blockSize,
        useHarrisDetector,
        k);

    /// Draw corners detected
    cout << "** Number of corners detected: " << corners.size() << endl;
    int r = 10;
    for (size_t i = 0; i < corners.size(); i++)
    {
        circle(copy, corners[i], r, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), -1, 8, 0);
    }

    /// Show what you got
    imshow("task 2", copy);
}