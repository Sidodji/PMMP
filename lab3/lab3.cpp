#include <opencv2\opencv.hpp>
using namespace cv;
using namespace std;

Mat frame, src_gray, src, detected_edges, img1, img2, img3;
Mat  grad_x, grad_y, grad_xy;
const int max_lowThreshold = 300;
const int ratio = 3;
const int kernel_size = 3;
const int max_thresh = 255;
int lowThreshold = 30;
const char* window_name = "Keni";

static void CannyThreshold(int, void*)
{
	blur(src_gray, detected_edges, Size(3, 3));
	Canny(detected_edges, detected_edges, lowThreshold, lowThreshold * 3, kernel_size);
	img3 = Scalar::all(0);
	frame.copyTo(img3, detected_edges);
	imshow(window_name, img3);
}

int main(int argc, char* argv[])
{
	namedWindow(window_name, WINDOW_AUTOSIZE);
	// видео с камеры
	VideoCapture capture(0);
	// проверка корректности открытия
	if (!capture.isOpened()) {
		cerr << "Unable to open: " << endl;
		return 0;
	}

	while (true) {
		// получение кадра видеопотока
		capture >> frame;
		//сглаживание
		GaussianBlur(frame, img1, Size(5, 5), 0, 0, BORDER_DEFAULT);

		cvtColor(img1, src_gray, COLOR_BGR2GRAY);
#pragma region Sobel
		Sobel(src_gray, grad_x, CV_16S, 1, 0, 3); // по Ox
		Sobel(src_gray, grad_y, CV_16S, 0, 1, 3); // по Oy
		// преобразование градиентов в 8-битные
		convertScaleAbs(grad_x, grad_x); // может быть отрицательным, принимать абсолютное значение, чтобы обеспечить отображение
		convertScaleAbs(grad_y, grad_y);
		// поэлементное вычисление взвешенной суммы двух массивов
		addWeighted(grad_x, 0.5, grad_y, 0.5, 0, img1); // смешать градиенты x, y

#pragma endregion Sobel
#pragma region Laplas
		Laplacian(src_gray, img2, CV_16S, 3, 1, 0, BORDER_DEFAULT);

		// converting back to CV_8U
		convertScaleAbs(img2, img2);

#pragma endregion Laplas
#pragma region Kenni
		
		//createTrackbar("Min_Threshold:", "Keni", &lowThreshold, max_lowThreshold, CannyThreshold);
		CannyThreshold(0, 0);
#pragma endregion Kenni
		// вывод кадра
		imshow("Sobel", img1);
		imshow("Laplas", img2);
		// условие для выхода из цикла 17
		
		int keyboard = waitKey(30);
		if (keyboard == 'q' || keyboard == 27)
			break;

	}
	return 0;
}

