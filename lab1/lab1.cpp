#include <opencv2\opencv.hpp>
using namespace cv;
#include <iostream>

void gist(Mat gray, std::string gHist_name) {
	Mat  gHist;
	// количество бинов гистограммы
	int kBins = 256;
	// интервал изменения значений бинов
	float range[] = { 0.0f, 256.0f };
	const float* histRange = { range };
	// равномерное распределение интервала по бинам
	bool uniform = true;
	// запрет очищения перед вычислением гистограммы
	bool accumulate = false;
	// размеры для отображения гистограммы
	int histWidth = 512, histHeight = 400;
	// количество пикселей на бин
	int binWidth = cvRound((double)histWidth / kBins);
	// вычисление гистограммы 
	calcHist(&gray, 1, 0, Mat(), gHist, 1, &kBins,
		&histRange, uniform, accumulate
	);

	double maxVal = 0;
	minMaxLoc(gHist, 0, &maxVal, 0, 0);

	int xscale = 10;
	int yscale = 10;

	cv::Mat hist_image;
	hist_image = cv::Mat::zeros(256, kBins * xscale, CV_8UC1);

	for (int s = 0; s < kBins; s++)
	{
		float binVal = gHist.at<float>(s, 0);
		int intensity = cvRound(binVal * 255 / maxVal);

		rectangle(hist_image, cv::Point(s * xscale, hist_image.rows),
			cv::Point((s + 1) * xscale - 1, hist_image.rows - intensity),
			cv::Scalar::all(255), 1);
	}

	namedWindow(gHist_name, WINDOW_AUTOSIZE);
	imshow(gHist_name, hist_image);
}

int main()
{
	Mat img, img1, img2;
#pragma region task2
	img = imread("D:/univer/PMiMP/images/eagle_head.jpg", IMREAD_COLOR);
	img1 = imread("D:/univer/PMiMP/images/eagle_head.jpg", IMREAD_GRAYSCALE);
	threshold(img1, img2, 120, 255, THRESH_BINARY);

	namedWindow("color", WINDOW_AUTOSIZE);
	namedWindow("grayscale", WINDOW_AUTOSIZE);
	namedWindow("binary", WINDOW_AUTOSIZE);

	imshow("color", img);
	imshow("grayscale", img1);
	imshow("binary", img2);
#pragma endregion task2

#pragma region task3
	imwrite("colore.jpg", img);
	imwrite("grayscale.jpg", img1);
	imwrite("binary.jpg", img2);
#pragma endregion task3

#pragma region task4
	Mat gray, eHist, img4;
	img4 = imread("D:/univer/PMiMP/images/eagle_head.jpg", IMREAD_COLOR);
	cvtColor(img4, gray, COLOR_BGR2GRAY);//конвертация изображения

	gist(gray, "color_histogram");
	namedWindow("color_img", WINDOW_AUTOSIZE);
	imshow("color_img", img4);

	equalizeHist(gray, eHist);
	gist(eHist, "equalized_histogram");
	imshow("equalized", eHist);
#pragma endregion task4

	waitKey(0);
	destroyAllWindows();
	return 0;
}