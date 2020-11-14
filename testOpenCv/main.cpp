//
//  main.cpp
//  testOpenCv
//
//  Created by An on 10/24/20.
//

#include <iostream>
#include <string>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"

using namespace cv;
using namespace std;

const int slider_max = 100;
int slider = 0;
double alpha;
double beta;

Mat image;
Mat grayImage;  // turn image into grayscale
Mat dst;   // change image brightness

Mat getGrayScaleImage(const Mat& image);
void changeBrightness(int pos ,void*);
void changeContrast(int pos, void*);
void averageFilterOperator(int pos, void*);
void gaussianFilterOperator(int pos, void*);
void detectEdge(Mat &srcImage, Mat &dstImage);

// MARK:- MAIN
int main(int argc, char *argv[]) {
    
//    image = imread("/Users/apple/Downloads/thisWall/0ottnoov9lo51.jpg");
    // create  and  set  the  windowname
    namedWindow("Show_Image");
    
    image = imread("/Users/apple/Documents/Stduy/openCv/testOpenCv/ic2.tif");
    if(argv[1] != NULL) {
        if (strcmp(argv[1], "-rgb2gray")) {
            grayImage = getGrayScaleImage(image);
            imshow("Show_Image", grayImage);
        }
        
        if(strcmp(argv[1], "-brightness")) {
            slider = 0;
            createTrackbar("Brightness","Show_Image", &slider , slider_max,changeBrightness);
            changeBrightness(slider, 0);
        }
        
        if(strcmp(argv[1], "-contrast")) {
            createTrackbar("Contrast","Show_Image", &slider , slider_max, changeContrast);
            slider = 1;
            changeContrast(slider, 0);
        }
        
        if(strcmp(argv[1], "-avg")) {
            createTrackbar("Average filter","Show_Image", &slider , slider_max, averageFilterOperator);
            slider = 3;
            averageFilterOperator(slider, 0);
        }
        
        if(strcmp(argv[1], "-gauss")) {
            createTrackbar("Guassian","Show_Image", &slider , slider_max, gaussianFilterOperator);
            slider = 3;
            gaussianFilterOperator(slider, 0);
        }
    }
    
//    if(strcmp(argv[1], "denoise")) {
        slider = 3;
        gaussianFilterOperator(slider, 0);
    //    create gray scale image
        grayImage = getGrayScaleImage(image);
        detectEdge(grayImage, dst);
//    }
    
    // close  the window
    waitKey (0);
    return 0;
}

// MARK: GRAYSCALE
Mat getGrayScaleImage(const Mat &image) {
    int width = image.cols , height = image.rows;
    unsigned long widthStep = image.step [0];
    unsigned long nChannels = image.step [1];
    Mat newImage = Mat(height, width, CV_8UC1);
    uchar* pData = (uchar *) image.data;
    uchar* nData = (uchar *) newImage.data;
    for(int y = 0; y < height; y++, pData  +=  widthStep, nData+=newImage.step[0]) {
        uchar* pRow = pData;
        uchar* nRow = nData;
        for(int x = 0; x < width; x++, pRow +=  nChannels, nRow+=newImage.step[1]) {
            uchar gray = (pRow[0] + pRow[1] + pRow[2])/3;
            nRow[0] = gray;
        }
    }
    return newImage;
}

// MARK: - CHANGE BRIGHTNESS
void changeBrightness(int pos ,void*){
    beta = (double) slider;
    dst = image.clone();
    int width = dst.cols , height = dst.rows;
    unsigned long widthStep = dst.step [0];
    unsigned long nChannels = dst.step [1];
    uchar* pData = (uchar *) dst.data;
    for(int y = 0; y < height; y++, pData  +=  widthStep) {
        uchar* pRow = pData;
        for(int x = 0; x < width; x++, pRow +=  nChannels) {
            pRow[0] = pRow[0] + beta > 255 ? 255 : pRow[0]+beta;
            pRow[1] = pRow[1] + beta > 255 ? 255 : pRow[1]+beta;
            pRow[2] = pRow[2] + beta > 255 ? 255 : pRow[2]+beta;
        }
    }
    imshow("Show_Image", dst);
}

// MARK: - CHANGE CONSTRAST
void changeContrast(int pos, void*) {
    alpha = (double) slider/slider_max;
    dst = image.clone();
    int width = dst.cols , height = dst.rows;
    unsigned long widthStep = dst.step [0];
    unsigned long nChannels = dst.step [1];
    uchar* pData = (uchar *) dst.data;
    for(int y = 0; y < height; y++, pData  +=  widthStep) {
        uchar* pRow = pData;
        for(int x = 0; x < width; x++, pRow +=  nChannels) {
            pRow[0] = pRow[0]*alpha >= 255 ? 255 : pRow[0]*alpha;
            pRow[1] = pRow[1]*alpha >= 255 ? 255 : pRow[1]*alpha;
            pRow[2] = pRow[2]*alpha >= 255 ? 255 : pRow[2]*alpha;
        }
    }
    imshow("Show_Image", dst);
}

// MARK: - AVERAGE FILTER
void averageFilterOperator(int pos, void*) {
    if (image.data == NULL || image.rows <= 0 || image.cols <= 0)
        return ;

    int width = image.cols, height = image.rows;

    dst = cv::Mat(height, width, image.type());

    unsigned long widthStep = image.step[0];
    unsigned long nChannels = image.step[1];
    

    int kHalfSize = slider / 2;
    
    vector<unsigned long> offsets;
    for (int y = -kHalfSize; y <= kHalfSize; y++)
        for (int x = -kHalfSize; x <= kHalfSize; x++)
            offsets.push_back(y*widthStep + x*nChannels);

    int xStart = kHalfSize, yStart = kHalfSize;
    int xEnd = width - kHalfSize - 1, yEnd = height - kHalfSize - 1;

    const uchar* pSrcData = (uchar*)image.data + yStart*widthStep + xStart*nChannels;
    uchar* pDstData = (uchar*)dst.data + yStart*widthStep + xStart*nChannels;
    
    unsigned long n = offsets.size();

    for (int y = yStart; y <= yEnd; y++, pSrcData += widthStep, pDstData += widthStep) {
        const uchar* pSrcRow = pSrcData;
        uchar* pDstRow = pDstData;

        for (int x = xStart; x <= xEnd; x++, pSrcRow += nChannels, pDstRow += nChannels) {
            for (int i = 0; i < nChannels; i++) {
                int avg = 0;
                for (int k = 0; k < n; k++)
                    avg += pSrcRow[i + offsets[k]];
                pDstRow[i] = (uchar)(avg / n);
            }
        }
    }
    
    imshow("Show_Image", dst);
}

// MARK:- Guassian Filter
void gaussianFilterOperator(int pos, void*) {
    if (image.data == NULL || image.rows <= 0 || image.cols <= 0)
        return ;

    int width = image.cols, height = image.rows;

    dst = Mat(height, width, image.type());

    unsigned long widthStep = image.step[0];
    unsigned long nChannels = image.step[1];
    
    float sigma = 1.4;
    int kHalfSize = slider / 2;
    
    vector<tuple<unsigned long, unsigned long>> offsets;
    float gauSum = 0;
    for (int y = -kHalfSize; y <= kHalfSize; y++) {
        for (int x = -kHalfSize; x <= kHalfSize; x++) {
            float gaus = (1/(2*M_PI*sigma*sigma))*2.71828182846 - ((x*x+y*y)/(2*sigma*sigma));
            gauSum += gaus;
            offsets.push_back(make_tuple(y*widthStep + x*nChannels, gaus));
        }
    }

    int xStart = kHalfSize, yStart = kHalfSize;
    int xEnd = width - kHalfSize - 1, yEnd = height - kHalfSize - 1;

    const uchar* pSrcData = (uchar*)image.data + yStart*widthStep + xStart*nChannels;
    uchar* pDstData = (uchar*)dst.data + yStart*widthStep + xStart*nChannels;
    
    unsigned long n = offsets.size();
    for (int y = yStart; y <= yEnd; y++, pSrcData += widthStep, pDstData += widthStep) {
        const uchar* pSrcRow = pSrcData;
        uchar* pDstRow = pDstData;
        for (int x = xStart; x <= xEnd; x++, pSrcRow += nChannels, pDstRow += nChannels) {
            for (int i = 0; i < nChannels; i++) {
                int avg = 0;
                for (int k = 0; k < n; k++)
                    avg += ((get<1>(offsets[1]) * pSrcRow[i + get<0>(offsets[k])]) / gauSum);
                pDstRow[i] = (uchar)(avg / n);
            }
        }
    }
    
    imshow("Show_Image", dst);
}

//MARK:- detect edge
void detectEdge(Mat &srcImage, Mat &dstImage) {
    if (srcImage.data == NULL || srcImage.rows <= 0 || srcImage.cols <= 0)
        return ;
    
    dstImage = Mat(srcImage.size(), srcImage.type());
    int width = srcImage.cols, height = srcImage.rows;
    int widthStep = srcImage.step[0];
    int nChannels = srcImage.step[1];
    Mat gradientX = Mat(srcImage.size(), srcImage.type());
    Mat gradientY = Mat(srcImage.size(), srcImage.type());
    
    int offsets[9] = {
        -widthStep-1, -widthStep, -widthStep+1,
        -1, 0, 1,
        widthStep-1, widthStep, widthStep+1,
    };
    
    int kernelX[9] = {-1, 0, 1,
                        -2, 0, 2,
                        -1, 0 ,1
    };
    
    int kernelY[9] = { -1, -2, -1,
                        0, 0, 0,
                        1, 2, 1
    };
    
    int xStart = 1, yStart = 1;
    int xEnd = width  - 1, yEnd = height - 1;

    const uchar* pSrcData = (uchar*)srcImage.data + yStart*widthStep + xStart*nChannels;
    uchar* pDstData = (uchar*)dstImage.data + yStart*widthStep + xStart*nChannels;

    unsigned long n = 9;

    for (int y = yStart; y <= yEnd; y++, pSrcData += widthStep, pDstData += widthStep) {
        const uchar* pSrcRow = pSrcData;
        uchar* pDstRow = pDstData;
        for (int x = xStart; x <= xEnd; x++, pSrcRow += nChannels, pDstRow += nChannels) {
            for (int i = 0; i < nChannels; i++) {
                int gx = 0, gy = 0;
                for (int k = 0; k < n; k++) {
                    gx += kernelX[k] * pSrcRow[i + offsets[k]];
                    gy += kernelY[k] * pSrcRow[i + offsets[k]];
                }
                int g = sqrt(gx*gx + gy*gy);
                if(g>80) {
                    pDstRow[0] = 255;
                }
            }
        }
    }
    imshow("Show_Image", dstImage);
}
