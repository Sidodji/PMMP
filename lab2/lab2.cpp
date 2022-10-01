#include <opencv2\opencv.hpp>
using namespace cv;
#include <iostream>

int main()
{
	Mat img, img1, img2, img3, img4, img5, element;
#pragma region task2
	img = imread("D:/univer/PMiMP/images/stonks.jpg", IMREAD_COLOR);
	threshold(img1, img2, 120, 255, THRESH_BINARY);

	namedWindow("color", WINDOW_AUTOSIZE);
	imshow("image", img);

	const float kernelData[] = { -1.0f, -1.0f, -1.0f,
		 -1.0f, 9.0f, -1.0f,
		 -1.0f, -1.0f, -1.0f };
	const Mat kernel(3, 3, CV_32FC1, (float*)kernelData);
	filter2D(img, img1, -1, kernel);

	namedWindow("svirtka", WINDOW_AUTOSIZE);
	imshow("svirtka", img1);
#pragma endregion task2
#pragma region task3
	Point anchor = Point(-1, -1);

	blur(img1, img2, Size(5, 5), anchor, BORDER_DEFAULT);

	GaussianBlur(img2, img2, Size(5, 5), 0, BORDER_DEFAULT);

	medianBlur(img2, img2, 15);

	namedWindow("blur", WINDOW_AUTOSIZE);

	imshow("blur", img2);
#pragma endregion task3
#pragma region task4	
	erode(img, img3, element); // вычисление эрозии
	dilate(img, img4, element); // вычисление дилатации

	img5 = img - img3; //разность изображений

	namedWindow("erode", WINDOW_AUTOSIZE);
	imshow("erode", img3);

	namedWindow("dilate", WINDOW_AUTOSIZE);
	imshow("dilate", img4);

	namedWindow("raznost", WINDOW_AUTOSIZE);
	imshow("raznost", img5);
#pragma endregion task4
	waitKey(0);
	destroyAllWindows();
	return 0;
}