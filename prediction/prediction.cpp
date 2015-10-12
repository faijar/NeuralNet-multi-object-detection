#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ml/ml.hpp"    
#include <sstream>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;
int thresh = 250;

#define ATTRIBUTES 875  //Number of pixels per sample.16X16
#define CLASSES 6

int main(int argc, char** argv)
{
	char name[2000];
	int y = 1;
	int ys = 1;
	int ys2 = 1;
	
	const String file_name = "model87.xml";
	//cv::Ptr<cv::ml::ANN_MLP> nnetwork = ml::StatModel::load<ml::ANN_MLP>("model.xml");
	Ptr<ml::ANN_MLP> nnetwork = ml::ANN_MLP::create();
	cv::FileStorage read("model.xml", cv::FileStorage::READ);
        nnetwork->read(read.root());	

	//while (1){

		//sprintf(name, "images/%04d.png", y);
		//Mat src = imread(name, 1);
		Mat src = imread( argv[1], 1 );
		Mat disp;
		src.copyTo(disp);
		if (!src.data)                              // Check for invalid input
		{
			cout << y;
			cout << "Could not open or find the image" << std::endl;
			return -1;
		}
		int largest_area = 0;
		int largest_contour_index = 0;
		Rect bounding_rect;
		Rect bounding_rect2;
		Mat canny_output;
		Mat canny_output2;
		//Mat src = imread("img (22).png"); //Load source image
		Mat thr(src.rows, src.cols, CV_8UC1);
		Mat thr1(src.rows, src.cols, CV_8UC1);
		Mat dst(src.rows, src.cols, CV_8UC1, Scalar::all(0));
		
		cvtColor(src, thr, CV_BGR2GRAY); //Convert to gray
		//dilate(thr, thr, )
		equalizeHist(thr, thr);

		int histSize = 256; // bin size
		float range[] = { 0, 256 };
		const float* histRange = { range };
		int nc = thr.channels();
		bool uniform = true;
		bool accumulate = false;
		int hmax = 0;
		int havg;
		Mat hist;

		int channels[] = { 0 };

		/// Compute the histograms:
		calcHist(&thr, 1, channels, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
		
		for (int j = 0; j < histSize - 1; j++){
			hmax = hmax+hist.at<int>(j);
		}
		havg = hmax/(histSize-1);	
		medianBlur(thr, thr, 5);
		
	   
	//	morphologyEx(thr, thr, MORPH_CLOSE, cv::noArray(), cv::Point(-1, -1), 2);
	//	threshold(thr, thr1, 86, 255, THRESH_BINARY); //Threshold the gray
		threshold(thr, thr, havg-20 , 255, THRESH_BINARY | THRESH_OTSU);
		//adaptiveThreshold(thr, thr, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY,11, 3);
		imshow("threashold", thr);
		vector<vector<Point> > contours; // Vector for storing contour
		vector<Vec4i> hierarchy;
		//Canny(thr, thr, thresh, thresh * 2, 3);
		findContours(thr, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE); // Find the contours in the image

		vector<vector<Point> > contours2; // Vector for storing contour
		vector<Vec4i> hierarchy2;
		//Canny(thr1, canny_output2, thresh, thresh * 2, 3);
		//findContours(thr1, contours2, hierarchy2, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE); // Find the contours in the image
		std::stringstream ss;
		ss << ys;		
		String is = ss.str();
		//String is2 = totring(ys2);
		for (int i = 0; i < contours.size(); i++) // iterate through each contour. 
		{
			//if (hierarchy[i][2] >= 0){
				double a = contourArea(contours[i], false);  //  Find the area of contour
				if (a > 250){

					//largest_area = a;
					//largest_contour_index = i;                //Store the index of largest contour
					bounding_rect = boundingRect(contours[i]); // Find the bounding rectangle for biggest contour
					
					//drawContours(src, contours, 0, Scalar(100, 255, 100), 1, 8, hierarchy);
					Mat imgSrc1 = src(bounding_rect);
					
					
					//pre-processing predict
					cvtColor(imgSrc1, imgSrc1, CV_BGR2GRAY);
					equalizeHist(imgSrc1, imgSrc1);
					medianBlur(imgSrc1, imgSrc1, 5);
					threshold(imgSrc1, imgSrc1, havg-20 , 255, THRESH_BINARY | THRESH_OTSU);
					//adaptiveThreshold(imgSrc1, imgSrc1, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 3, 0);
					Size size(35,25);//the dst image size,e.g.100x100
					resize(imgSrc1,imgSrc1,size);//resize image

					//Neural network predict
					cv::Mat data(1, ATTRIBUTES, CV_32F);
					Mat dat = imgSrc1.reshape(0,1);
					dat.convertTo(dat, CV_32F);
					//data = dat.clone();
					//imgSrc1.copyTo(data(Rect(0, 0, imgSrc1.cols, imgSrc1.rows)));
					//cout<< data.cols;
					int maxIndex = 0;
					cv::Mat classOut(1, CLASSES, CV_32F);
					nnetwork->predict(data, classOut);
					float value;
					float maxValue = classOut.at<float>(0, 0);
					for (int index = 1; index<CLASSES; index++)
					{
						value = classOut.at<float>(0, index);
						if (value>maxValue)
						{
							maxValue = value;
							maxIndex = index;
						}
					}
					cout<<"prediction class index :" <<maxIndex <<"\n";
					if(maxIndex == 0){
						rectangle(src, bounding_rect, Scalar(200, 255, 100), 2, 8, 0);//button color cyan
					}else if(maxIndex == 1){
						rectangle(src, bounding_rect, Scalar(0, 255, 0), 2, 8, 0); //text color green
					}else if(maxIndex == 2){
						rectangle(src, bounding_rect, Scalar(255, 0, 0), 2, 8, 0);//image color blue
					}else if(maxIndex == 3){
						rectangle(src, bounding_rect, Scalar(0, 0, 255), 2, 8, 0);//datepicker color red
					}else if(maxIndex == 4){
						rectangle(src, bounding_rect, Scalar(100, 255, 255), 2, 8, 0);//dropdown color yellow
					}else if(maxIndex == 5){
						rectangle(src, bounding_rect, Scalar(255, 255, 0), 2, 8, 0);// tabs color light blue
					}else if(maxIndex == 6){
						rectangle(src, bounding_rect, Scalar(200, 0, 200), 2, 8, 0);// other
					}

					
					//imwrite("data/data1-"+ is + ".png", src);

					
				}
			//}
			//bounding_rect = boundingRect(contours[i]);

		}
		/*
		for (int i = 0; i < contours2.size(); i++) // iterate through each contour. 
		{

			double a = contourArea(contours2[i], false);  //  Find the area of contour
			if (a > 150){
				//largest_area = a;
				//largest_contour_index = i;                //Store the index of largest contour
				bounding_rect2 = boundingRect(contours2[i]); // Find the bounding rectangle for biggest contour
				rectangle(src, bounding_rect2, Scalar(250, 255, 100), 2, 8, 0);
				//Mat imgSrc2 = src(bounding_rect2);
				//imwrite("data/data2-" + is2 + ".png", imgSrc2);
				ys2++;
			}

			//bounding_rect = boundingRect(contours[i]);

		}
		*/
		//Scalar color(255, 255, 255);
		//drawContours(dst, contours, largest_contour_index, color, CV_FILLED, 8, hierarchy); // Draw the largest contour using previously stored index.
		ys++;
		imshow("src", src);
		//imwrite("contour result 1391076a-ed94-4060-8bb7-102e8597f8fe.png", src);
		y++;
		//waitKey();
	//}
	//imshow("largest Contour", dst);
	waitKey();
}
