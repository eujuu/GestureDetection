#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2\opencv.hpp>
#include <iostream>
#include <string>
#include <Windows.h>
#include <winuser.h>
#include <stdlib.h> 
#include <string.h> 
#include <tchar.h>
#include <Windows.h>

using namespace cv;
using namespace std;

void salt(Mat &img, int n);

int main(int argc, char** argv)
{
	VideoCapture cap(0); // open the default camera
	if (!cap.isOpened()) // check if we succeeded
		return -1;
	namedWindow("Original", WINDOW_AUTOSIZE);
	char TrackbarName[50];
	Mat prof_img = imread("profe.bmp");
	Mat our_img = imread("522.bmp");
	for (;;) {
		Mat origin_image, ycrcb, skin, clonee;
		cap >> origin_image;
		cap >> clonee;
		cvtColor(origin_image, ycrcb, CV_BGR2YCrCb);
		inRange(ycrcb, Scalar(0, 133, 77), Scalar(255, 173, 127), skin);

		//-----전처리-----//
		Mat mask = getStructuringElement(MORPH_RECT, Size(5, 5), Point(-1, -1)); //커널 매트릭스 생성
		morphologyEx(skin, skin, MorphTypes::MORPH_ERODE, mask); //침식
		morphologyEx(skin, skin, MorphTypes::MORPH_OPEN, mask); //열기
		morphologyEx(skin, skin, MorphTypes::MORPH_CLOSE, mask);//닫기
		medianBlur(skin, skin, 3);//미디언필터링


								  //-----가장 큰 영역 찾아서 컨투어하기-----//
		int largest_contour_index = 0;
		//Rect bounding_rect;
		vector<vector<Point>> contours; // Vector for storing contour
		vector<Vec4i> hierarchy;
		findContours(skin, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

		Mat maskk = Mat::zeros(origin_image.size(), CV_8U);
		// iterate through each contour.
		int largest_area = 50;   //제일 큰 area 찾기
		for (int i = 0; i < contours.size(); i++)
		{
			//  Find the area of contour
			double a = contourArea(contours[i], false); //
			if (a > largest_area) {
				largest_area = a;
				//cout << i << " area  " << a << endl;
				// Store the index of largest contour
				largest_contour_index = i;
				// Find the bounding rectangle for biggest contour
				//bounding_rect = boundingRect(contours[i]);
			}
		}
		Scalar color(255, 255, 255);
		drawContours(maskk, contours, largest_contour_index, color, -1, 8, hierarchy); 

		erode(maskk, maskk, Mat(3, 3, CV_8U, Scalar(1)), Point(-1, -1), 2);

		Mat dst; // 거리 변환 행렬을 저장할 변수
		distanceTransform(maskk, dst, CV_DIST_L2, 5);   //입력 영상의 각 픽셀에서 픽셀값이 0인 픽셀까지 가장 가까운 거리를 계산
		double radius;
		int maxldx[2];
		minMaxIdx(dst, NULL, &radius, NULL, maxldx, maskk);
	
		Point center = Point(maxldx[1], maxldx[0]);
		Point client = GetClientRect();
		SetCursorPos(center.x+client.x, center.y+client.y);

		circle(origin_image, center, 2, Scalar(0, 255, 0), -1);   //손바닥 중심점
		circle(origin_image, center, (int)(radius + 50), Scalar(255, 0, 0), 2);

		Mat circled = Mat::zeros(origin_image.size(), CV_8U);
		circle(circled, center, (int)(radius + 65), Scalar(255, 255, 255), -1);
		maskk = maskk - circled;

		Mat img_labels, stats, centroids;   //stats : 레이블링된 이미지 배열, centroids : 레이블링된 이미지의 중심좌표
		int numOfLables = connectedComponentsWithStats(maskk, img_labels, stats, centroids, 8, CV_32S);//skin: input/ img_labels= output/

		int num = numOfLables - 1;
		sprintf_s(TrackbarName, "%d", num);
		putText(origin_image, TrackbarName, Point(50, 100), FONT_HERSHEY_SIMPLEX, 3, Scalar(0, 0, 0), 5, 8);

		if (num == 1) {
			Mat newonew = our_img.clone();

			addWeighted(our_img, 0.1, prof_img, 0.1, 0, newonew);
			waitKey(1);
			addWeighted(our_img, 0.2, prof_img, 0.2, 0, newonew);
			waitKey(1);
			addWeighted(our_img, 0.3, prof_img, 0.3, 0, newonew);
			waitKey(1);
			addWeighted(our_img, 0.4, prof_img, 0.4, 0, newonew);
			waitKey(1);
			addWeighted(our_img, 0.5, prof_img, 0.5, 0, newonew);
			waitKey(1);
			namedWindow("ROI");
			imshow("ROI", newonew);
		}
		if (num == 2) {
			namedWindow("professor");
			imshow("professor", prof_img);
			namedWindow("our");
			imshow("our", our_img);
			waitKey(1);
		}
		if (num == 3) {
			Mat inv;
			flip(origin_image, inv, 0);
			namedWindow("Inverse");
			imshow("Inverse", inv);

		}
		if (num == 4) {
			salt(clonee, 3000);
			namedWindow("Salt");
			imshow("Salt", clonee);
		}
		if (num == 5) {
			imwrite("capture.png", origin_image);
		}

		namedWindow("Binary window", CV_WINDOW_AUTOSIZE);
		imshow("Binary window", maskk);
		namedWindow("Display window", CV_WINDOW_AUTOSIZE);
		imshow("Display window", origin_image);

		if (waitKey(30) >= 0)

			break;
	}

	int c = waitKey(10);
	if (c == 27) {
		destroyAllWindows();
	}

	waitKey();
	return 0;
}

void salt(Mat &img, int n)
{
	for (int k = 0; k<n; k++) {
		int i = rand() % img.cols;
		int j = rand() % img.rows;
		if (img.channels() == 1) { // Gray scale image
			img.at<uchar>(j, i) = 255;
		}
		else if (img.channels() == 3) { // Color image
			img.at<Vec3b>(j, i)[0] = 255;
			img.at<Vec3b>(j, i)[1] = 255;
			img.at<Vec3b>(j, i)[2] = 255;
		}
	}
}