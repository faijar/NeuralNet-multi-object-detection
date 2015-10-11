#include <iostream>
#include "opencv2/opencv.hpp"
#include <stdio.h>
#include <fstream>
using namespace std;
using namespace cv;


Mat src,img,ROI;
Rect cropRect(0,0,0,0);
 Point P1(0,0);
 Point P2(0,0);

const char* winName="Crop Image";
bool clicked=false;
int i=2030;
char imgName[15];


void checkBoundary(){
       //check croping rectangle exceed image boundary
       if(cropRect.width>img.cols-cropRect.x)
         cropRect.width=img.cols-cropRect.x;

       if(cropRect.height>img.rows-cropRect.y)
         cropRect.height=img.rows-cropRect.y;

        if(cropRect.x<0)
         cropRect.x=0;

       if(cropRect.y<0)
         cropRect.height=0;
}

void showImage(){
    img=src.clone();
    checkBoundary();
    if(cropRect.width>0&&cropRect.height>0){
        ROI=src(cropRect);
        imshow("cropped",ROI);
    }

    rectangle(img, cropRect, Scalar(0,255,0), 1, 8, 0 );
    imshow(winName,img);
}


void onMouse( int event, int x, int y, int f, void* ){


    switch(event){

        case  CV_EVENT_LBUTTONDOWN  :
                                        clicked=true;

                                        P1.x=x;
                                        P1.y=y;
                                        P2.x=x;
                                        P2.y=y;
                                        break;

        case  CV_EVENT_LBUTTONUP    :
                                        P2.x=x;
                                        P2.y=y;
                                        clicked=false;
                                        break;

        case  CV_EVENT_MOUSEMOVE    :
                                        if(clicked){
                                        P2.x=x;
                                        P2.y=y;
                                        }
                                        break;

        default                     :   break;


    }


    if(clicked){
     if(P1.x>P2.x){ cropRect.x=P2.x;
                       cropRect.width=P1.x-P2.x; }
        else {         cropRect.x=P1.x;
                       cropRect.width=P2.x-P1.x; }

        if(P1.y>P2.y){ cropRect.y=P2.y;
                       cropRect.height=P1.y-P2.y; }
        else {         cropRect.y=P1.y;
                       cropRect.height=P2.y-P1.y; }

    }


showImage();


}

void writeMatToFile(cv::Mat& mats, const char* filename, int value)
{		
		Mat m;
		cvtColor(mats, m, CV_BGR2GRAY);
		equalizeHist(m, m);

		int histSize = 256; // bin size
		float range[] = { 0, 256 };
		const float* histRange = { range };
		int nc = m.channels();
		bool uniform = true;
		bool accumulate = false;
		int hmax = 0;
		int havg;
		Mat hist;

		int channels[] = { 0 };

		/// Compute the histograms:
		calcHist(&m, 1, channels, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
		
		for (int j = 0; j < histSize - 1; j++){
			hmax = hmax+hist.at<int>(j);
		}
		havg = hmax/(histSize-1);	
		medianBlur(m, m, 5);
		
	   
	//	morphologyEx(thr, thr, MORPH_CLOSE, cv::noArray(), cv::Point(-1, -1), 2);
	//	threshold(thr, thr1, 86, 255, THRESH_BINARY); //Threshold the gray
		threshold(m, m, havg-20 , 255, THRESH_BINARY | THRESH_OTSU);
		Size size(35,25);//the dst image size,e.g.100x100
		imshow("bin", m);
		resize(m,m,size);//resize image
    		ofstream fout(filename, ios::app);
	
    if(!fout)
    {
        cout<<"File Not Opened"<<endl;  return;
    }

    for(int i=0; i<m.rows; i++)
    {
        for(int j=0; j<m.cols; j++)
        {
	    int pix = m.at<uchar>(i,j);
            fout<< pix <<",";
        }
	
    }
     fout<< value;
    fout<<endl;
    fout.close();
}

int main()
{
    char name[2000];
	int y = 176;
	int ys = 1;
	int ys2 = 1;
    const char* filename = "data.txt";

    //src=imread("0002.png",1);

    namedWindow(winName,WINDOW_NORMAL);
    setMouseCallback(winName,onMouse,NULL );
    //imshow(winName,src);

    while(1){
	sprintf(name, "source/%04d.png", y);
        src = imread(name, 1);
    char c=waitKey();
    if(c=='s'&&ROI.data){
     sprintf(imgName,"%04d.png",i++);
     imwrite(imgName,ROI);
     cout<<"  Saved "<<imgName<<endl;
    }
    if(c=='0')
	{
 	writeMatToFile(ROI, filename, 0);
 	sprintf(imgName,"button/%04d.png",i++);
     	imwrite(imgName,ROI);
	cropRect.x=0;cropRect.y=0;cropRect.width=0;cropRect.height=0;
	}
    if(c=='1'){
	writeMatToFile(ROI, filename, 1);
 	sprintf(imgName,"text/%04d.png",i++);
     	imwrite(imgName,ROI);
	cropRect.x=0;cropRect.y=0;cropRect.width=0;cropRect.height=0;
	}
    if(c=='2'){
	writeMatToFile(ROI, filename, 2);
 	sprintf(imgName,"image/%04d.png",i++);
     	imwrite(imgName,ROI);
	cropRect.x=0;cropRect.y=0;cropRect.width=0;cropRect.height=0;
	}
    if(c=='3'){
	writeMatToFile(ROI, filename, 3);
 	sprintf(imgName,"datepicker/%04d.png",i++);
     	imwrite(imgName,ROI);
	cropRect.x=0;cropRect.y=0;cropRect.width=0;cropRect.height=0;
	}
    if(c=='4'){
	writeMatToFile(ROI, filename, 4);
 	sprintf(imgName,"dropdown/%04d.png",i++);
     	imwrite(imgName,ROI);
	cropRect.x=0;cropRect.y=0;cropRect.width=0;cropRect.height=0;
	}
    if(c=='5'){
	writeMatToFile(ROI, filename, 5);
 	sprintf(imgName,"todo/%04d.png",i++);
     	imwrite(imgName,ROI);
	cropRect.x=0;cropRect.y=0;cropRect.width=0;cropRect.height=0;
	}
    if(c=='6'){
	writeMatToFile(ROI, filename, 6);
 	sprintf(imgName,"other/%04d.png",i++);
     	imwrite(imgName,ROI);
	cropRect.x=0;cropRect.y=0;cropRect.width=0;cropRect.height=0;
	};

    if(c=='n') y++;
 

    if(c==27) break;
    if(c=='r') {cropRect.x=0;cropRect.y=0;cropRect.width=0;cropRect.height=0;}
    showImage();

    }


    return 0;
}
