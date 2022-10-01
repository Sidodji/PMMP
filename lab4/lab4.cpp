#include <opencv2\opencv.hpp>

using namespace cv;
using namespace std;

Mat src_gray, src,src2, src3, src_gray2, src_gray2_1, srcCanny2, src3Gray, srcCannyBGR_P, imgLine, srcCannyBGR, imgLineGray, srcCanny, imgHouht, srcCannyGray;
int thresh = 100;
RNG rng(12345);

void thresh_callback(int, void*);
void HouhtLine(int, void*);
void HouhtLineP(int, void*);
void Circle(int, void*);

int main(int argc, char* argv[])
{
    //CommandLineParser parser(argc, argv, "{@input | HappyFish.jpg | input image}");
    src = imread("D:/univer/PMMP/image/spun.jpg", IMREAD_COLOR);
    src2 = imread("D:/univer/PMMP/image/large_thumbnail.jpg", IMREAD_COLOR);
    src3 = imread("D:/univer/PMMP/image/Ball.jpg", IMREAD_COLOR);
    
    if (src.empty())
    {
        cout << "Could not open  or find the image!\n" << endl;
        cout << "Usage: " << argv[0] << " <Input image>" << endl;
        return -1;
    }

    cvtColor(src, src_gray, COLOR_BGR2GRAY);//gray
    cvtColor(src2, src_gray2, COLOR_BGR2GRAY);//gray
    src_gray2_1 = src_gray2.clone();
    //threshold(src_gray2, src_gray2, 120, 255, THRESH_BINARY);
    Canny(src2, srcCanny, 50, 200, 3);

    cvtColor(srcCanny, srcCannyBGR, COLOR_GRAY2BGR);//BGR
    srcCannyBGR_P = srcCannyBGR.clone();
    

    Mat imgLineGrayP = imgLineGray.clone();

    blur(src_gray, src_gray, Size(3, 3));
    const char* source_window = "Source";
    namedWindow(source_window);
    imshow(source_window, src);
    const int max_thresh = 255;
    createTrackbar("Canny thresh:", source_window, &thresh, max_thresh, thresh_callback);
    thresh_callback(0, 0);

    HouhtLine(0, 0);

    HouhtLineP(0, 0);

    cvtColor(src3, src3Gray, COLOR_BGR2GRAY);
    medianBlur(src3Gray, src3Gray, 5);
    Circle(0, 0);

    waitKey();
    return 0;
}

void thresh_callback(int, void*)
{
    Mat canny_output;
    Canny(src_gray, canny_output, thresh, thresh * 2);
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(canny_output, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
    for (size_t i = 0; i < contours.size(); i++)
    {
        Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        drawContours(drawing, contours, (int)i, color, 2, LINE_8, hierarchy, 0);
    }
    cout << contours.size();
    imshow("Contours", drawing);
}

void HouhtLine(int, void*)
{
    vector<Vec2f> lines; // will hold the results of the detection
    HoughLines(src_gray2, lines, 200, CV_PI / 180, 50, 50, 10); // runs the actual detection
    // Draw the lines
    for (size_t i = 0; i < lines.size(); i++)
    {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));
        line(src_gray2, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
    }

    imshow("Hought", src_gray2);
}

void HouhtLineP(int, void*)
{
    vector<Vec4i> linesP; // will hold the results of the detection
    HoughLinesP(src_gray2_1, linesP, 200, CV_PI / 180, 50, 50, 10); // runs the actual detection
    // Draw the lines
    for (size_t i = 0; i < linesP.size(); i++)
    {
        Vec4i l = linesP[i];
        line(src_gray2_1, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
    }

    imshow("Hought_P", src_gray2_1);
}

void Circle(int, void*)
{
    //src3Gray.rows / 16
    vector<Vec3f> circles;
    HoughCircles(src3Gray, circles, HOUGH_GRADIENT, 1,
        100,  // change this value to detect circles with different distances to each other
        100, 30, 30, 120 // change the last two parameters
        // (min_radius & max_radius) to detect larger circles
    );

    for (size_t i = 0; i < circles.size(); i++)
    {
        Vec3i c = circles[i];
        Point center = Point(c[0], c[1]);
        // circle center
        circle(src3, center, 1, Scalar(0, 100, 100), 2, LINE_AA);
        // circle outline
        int radius = c[2];
        circle(src3, center, radius, Scalar(255, 0, 255), 2, LINE_AA);
    }

    imshow("detected circles", src3);
}




