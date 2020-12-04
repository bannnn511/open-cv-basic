//
//  main.cpp
//  testOpenCv
//
//  Created by An on 10/24/20.
//

#include <math.h>

#include <iostream>
#include <string>

#include "opencv2/core/core.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"

using namespace cv;
using namespace std;

const int slider_max = 100;
int slider = 0;
double alpha;
double beta;

Mat image;
Mat grayImage;  // turn image into grayscale
Mat dst;        // change image brightness

Mat getGrayScaleImage(const Mat& image);
void changeBrightness(int pos, void*);
void changeContrast(int pos, void*);
void averageFilterOperator(int pos, void*);
void gaussianFilterOperator(int pos, void*);
void detectEdge(Mat& srcImage, Mat& dstImage);
void addNoise(Mat& srcImage);
void houghTransform(Mat& dst);

// MARK:- MAIN
int main(int argc, char* argv[]) {
  // image = imread("/Users/apple/Downloads/thisWall/0ottnoov9lo51.jpg");
  // create  and  set  the  windowname
  namedWindow("Show_Image");
  if (argv[1] != NULL) {
    // image = imread("/Users/apple/Documents/Stduy/openCv/testOpenCv/ic2.tif");
    image = imread(argv[2]);
    if (image.data == NULL) {
      cout << "Need image location" << endl;
    }
    if (strcmp(argv[1], "-rgb2gray") == 0) {
      grayImage = getGrayScaleImage(image);
      imshow("Show_Image", grayImage);
    } else if (strcmp(argv[1], "-brightness") == 0) {
      slider = 0;
      createTrackbar("Brightness", "Show_Image", &slider, slider_max,
                     changeBrightness);
      changeBrightness(slider, 0);
    } else if (strcmp(argv[1], "-contrast") == 0) {
      createTrackbar("Contrast", "Show_Image", &slider, slider_max,
                     changeContrast);
      slider = 1;
      changeContrast(slider, 0);
    } else if (strcmp(argv[1], "-avg") == 0) {
      createTrackbar("Average filter", "Show_Image", &slider, slider_max,
                     averageFilterOperator);
      slider = 3;
      averageFilterOperator(slider, 0);
    } else if (strcmp(argv[1], "-gauss") == 0) {
      createTrackbar("Gaussian", "Show_Image", &slider, slider_max,
                     gaussianFilterOperator);
      slider = 3;
      gaussianFilterOperator(slider, 0);
    } else if (strcmp(argv[1], "-addnoise") == 0) {
      image = getGrayScaleImage(image);
      addNoise(image);
    } else if (strcmp(argv[1], "-denoise") == 0) {
      slider = 5;
      image = getGrayScaleImage(image);
      averageFilterOperator(slider, 0);
    } else if (strcmp(argv[1], "-edges") == 0) {
      slider = 3;
      image = getGrayScaleImage(image);
      gaussianFilterOperator(slider, 0);
      detectEdge(image, dst);
      houghTransform(dst);
    }
  }

  // close  the window
  waitKey(0);
  return 0;
}

// MARK: GRAYSCALE
Mat getGrayScaleImage(const Mat& image) {
  int width = image.cols, height = image.rows;
  unsigned long widthStep = image.step[0];
  unsigned long nChannels = image.step[1];
  Mat newImage = Mat(height, width, CV_8UC1);
  uchar* pData = (uchar*)image.data;
  uchar* nData = (uchar*)newImage.data;
  for (int y = 0; y < height;
       y++, pData += widthStep, nData += newImage.step[0]) {
    uchar* pRow = pData;
    uchar* nRow = nData;
    for (int x = 0; x < width;
         x++, pRow += nChannels, nRow += newImage.step[1]) {
      uchar gray = (pRow[0] + pRow[1] + pRow[2]) / 3;
      nRow[0] = gray;
    }
  }
  return newImage;
}

// MARK: - CHANGE BRIGHTNESS
void changeBrightness(int pos, void*) {
  beta = (double)slider;
  dst = image.clone();
  int width = dst.cols, height = dst.rows;
  unsigned long widthStep = dst.step[0];
  unsigned long nChannels = dst.step[1];
  uchar* pData = (uchar*)dst.data;
  for (int y = 0; y < height; y++, pData += widthStep) {
    uchar* pRow = pData;
    for (int x = 0; x < width; x++, pRow += nChannels) {
      pRow[0] = pRow[0] + beta > 255 ? 255 : pRow[0] + beta;
      pRow[1] = pRow[1] + beta > 255 ? 255 : pRow[1] + beta;
      pRow[2] = pRow[2] + beta > 255 ? 255 : pRow[2] + beta;
    }
  }
  imshow("Show_Image", dst);
}

// MARK: - CHANGE CONSTRAST
void changeContrast(int pos, void*) {
  alpha = (double)slider / slider_max;
  dst = image.clone();
  int width = dst.cols, height = dst.rows;
  unsigned long widthStep = dst.step[0];
  unsigned long nChannels = dst.step[1];
  uchar* pData = (uchar*)dst.data;
  for (int y = 0; y < height; y++, pData += widthStep) {
    uchar* pRow = pData;
    for (int x = 0; x < width; x++, pRow += nChannels) {
      pRow[0] = pRow[0] * alpha >= 255 ? 255 : pRow[0] * alpha;
      pRow[1] = pRow[1] * alpha >= 255 ? 255 : pRow[1] * alpha;
      pRow[2] = pRow[2] * alpha >= 255 ? 255 : pRow[2] * alpha;
    }
  }
  imshow("Show_Image", dst);
}

// MARK: - AVERAGE FILTER
void averageFilterOperator(int pos, void*) {
  if (image.data == NULL || image.rows <= 0 || image.cols <= 0) return;

  int width = image.cols, height = image.rows;

  dst = cv::Mat(height, width, image.type());

  unsigned long widthStep = image.step[0];
  unsigned long nChannels = image.step[1];

  if (slider % 2 == 0) {
    slider++;
    slider = slider < 3 ? 3 : slider;
  }
  int kHalfSize = slider / 2;

  vector<unsigned long> offsets;
  for (int y = -kHalfSize; y <= kHalfSize; y++)
    for (int x = -kHalfSize; x <= kHalfSize; x++)
      offsets.push_back(y * widthStep + x * nChannels);

  int xStart = kHalfSize, yStart = kHalfSize;
  int xEnd = width - kHalfSize - 1, yEnd = height - kHalfSize - 1;

  const uchar* pSrcData =
      (uchar*)image.data + yStart * widthStep + xStart * nChannels;
  uchar* pDstData = (uchar*)dst.data + yStart * widthStep + xStart * nChannels;

  unsigned long n = offsets.size();

  for (int y = yStart; y <= yEnd;
       y++, pSrcData += widthStep, pDstData += widthStep) {
    const uchar* pSrcRow = pSrcData;
    uchar* pDstRow = pDstData;

    for (int x = xStart; x <= xEnd;
         x++, pSrcRow += nChannels, pDstRow += nChannels) {
      for (int i = 0; i < nChannels; i++) {
        int avg = 0;
        for (int k = 0; k < n; k++) avg += pSrcRow[i + offsets[k]];
        pDstRow[i] = (uchar)(avg / n);
      }
    }
  }

  imshow("Show_Image", dst);
}

// MARK:- Gaussian Filter
void gaussianFilterOperator(int pos, void*) {
  if (image.data == NULL || image.rows <= 0 || image.cols <= 0) return;

  int width = image.cols, height = image.rows;

  dst = Mat(height, width, image.type());

  unsigned long widthStep = image.step[0];
  unsigned long nChannels = image.step[1];

  float sigma = 1.8;
  if (slider % 2 == 0) {
    slider++;
    slider = slider < 3 ? 3 : slider;
  }
  int kHalfSize = slider / 2;

  vector<tuple<int, float>> offsets;
  float gauSum = 0;
  for (int y = -kHalfSize; y <= kHalfSize; y++) {
    for (int x = -kHalfSize; x <= kHalfSize; x++) {
      float gauss = (1 / (2 * M_PI * sigma * sigma)) *
                    exp(-((x * x + y * y) / (2 * sigma * sigma)));
      gauSum += gauss;
      offsets.push_back(make_tuple(y * widthStep + x * nChannels, gauss));
    }
  }

  int xStart = kHalfSize, yStart = kHalfSize;
  int xEnd = width - kHalfSize - 1, yEnd = height - kHalfSize - 1;

  const uchar* pSrcData =
      (uchar*)image.data + yStart * widthStep + xStart * nChannels;
  uchar* pDstData = (uchar*)dst.data + yStart * widthStep + xStart * nChannels;

  unsigned long n = offsets.size();
  for (int y = yStart; y <= yEnd;
       y++, pSrcData += widthStep, pDstData += widthStep) {
    const uchar* pSrcRow = pSrcData;
    uchar* pDstRow = pDstData;
    for (int x = xStart; x <= xEnd;
         x++, pSrcRow += nChannels, pDstRow += nChannels) {
      for (int i = 0; i < nChannels; i++) {
        float avg = 0;
        for (int k = 0; k < n; k++) {
          avg +=
              ((get<1>(offsets[k]) * pSrcRow[i + get<0>(offsets[k])]) / gauSum);
        }
        pDstRow[i] = (uchar)(avg);
      }
    }
  }

  imshow("Show_Image", dst);
}

// MARK:- detect edge
void detectEdge(Mat& srcImage, Mat& dstImage) {
  if (srcImage.data == NULL || srcImage.rows <= 0 || srcImage.cols <= 0) return;

  int highThreshold = 200, lowThreshold = 50;
  dstImage = Mat(srcImage.size(), srcImage.type());
  int width = srcImage.cols, height = srcImage.rows;
  int widthStep = srcImage.step[0];
  int nChannels = srcImage.step[1];
  Mat gradientX = Mat(srcImage.size(), srcImage.type());
  Mat gradientY = Mat(srcImage.size(), srcImage.type());

  int offsets[9] = {
      -widthStep - 1, -widthStep, -widthStep + 1, -1, 0, 1,
      widthStep - 1,  widthStep,  widthStep + 1,
  };

  int kernelX[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};

  int kernelY[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

  int xStart = 1, yStart = 1;
  int xEnd = width - 1, yEnd = height - 1;

  Mat gradientDirection = image.clone();  // image that save direction
  const uchar* pSrcData =
      (uchar*)srcImage.data + yStart * widthStep + xStart * nChannels;
  uchar* pDstData =
      (uchar*)dstImage.data + yStart * widthStep + xStart * nChannels;
  uchar* pDirData =
      (uchar*)gradientDirection.data + yStart * widthStep + xStart * nChannels;

  // kernel size
  unsigned long n = 9;

  // convolution step
  for (int y = yStart; y <= yEnd; y++, pSrcData += widthStep,
           pDstData += widthStep, pDirData += widthStep) {
    const uchar* pSrcRow = pSrcData;
    uchar* pDstRow = pDstData;
    uchar* pDirRow = pDirData;
    for (int x = xStart; x <= xEnd; x++, pSrcRow += nChannels,
             pDstRow += nChannels, pDirRow += nChannels) {
      for (int i = 0; i < nChannels; i++) {
        float gx = 0, gy = 0;
        for (int k = 0; k < n; k++) {
          gx += kernelX[k] * pSrcRow[i + offsets[k]];
          gy += kernelY[k] * pSrcRow[i + offsets[k]];
        }
        // edge gradient
        int g = sqrt(gx * gx + gy * gy);
        g = g > 255 ? 255 : g;
        // edge directions are classified into four group of angles 0, 45, 90,
        // 135 degrees
        int angle = atan(gy / gx);
        if (gx == 0) {
          angle = 90;
        }
        int theta = int(round(angle * (5.0 / M_PI) + 5)) % 5;
        theta = theta < 0 ? theta * -1 % 4 : theta % 4;
        // int theta = angle * 180 / M_PI;
        // if (g > 80) {
        //   pDstRow[0] = 255;
        // }
        pDstRow[0] = g;
        pDirRow[0] = theta;
      }
    }
  }

  // restart position
  widthStep = dstImage.step[0];
  pDstData = (uchar*)dstImage.data + yStart * widthStep + xStart * nChannels;
  pDirData =
      (uchar*)gradientDirection.data + yStart * widthStep + xStart * nChannels;

  // Non-maximum Suppression + Hysteresis Thresholding
  for (int y = yStart; y <= yEnd;
       y++, pDstData += widthStep, pDirData += widthStep) {
    uchar* pDstRow = pDstData;
    uchar* pDirRow = pDirData;
    for (int x = xStart; x <= xEnd;
         x++, pDstRow += nChannels, pDirRow += nChannels) {
      int pDstRowN;
      int pDstRowS;
      // cout << pDirRow[0] << endl;
      if (pDirRow[0] == 0) {
        pDstRowN = pDstRow[-1];
        pDstRowS = pDstRow[+1];
      } else if (pDirRow[0] == 1) {
        pDstRowN = pDstRow[-widthStep + 1];
        pDstRowS = pDstRow[+widthStep - 1];
      } else if (pDstRow[0] == 2) {
        pDstRowN = pDstRow[-widthStep];
        pDstRowS = pDstRow[+widthStep];
      } else if (pDstRow[0] == 3) {
        pDstRowN = pDstRow[-widthStep - 1];
        pDstRowS = pDstRow[+widthStep + 1];
      }

      if (pDstRow[0] < pDstRowN || pDstRow[0] < pDstRowS ||
          pDstRow[0] < lowThreshold) {
        pDstRow[0] = 0;
      }
      // if (pDstRow[0] < lowThreshold) {
      //   pDstRow[0] = 0;
      // }
      if (pDstRow[0] > lowThreshold && pDstRow[0] < highThreshold) {
        int nw = pDstRow[-widthStep - 1];
        int n = pDstRow[-widthStep];
        int ne = pDstRow[-widthStep + 1];
        int w = pDstRow[-1];
        int e = pDstRow[+1];
        int sw = pDstRow[+widthStep - 1];
        int s = pDstRow[+widthStep];
        int se = pDstRow[+widthStep + 1];
        if (nw > highThreshold || n > highThreshold || ne > highThreshold ||
            w > highThreshold || e > highThreshold || sw > highThreshold ||
            s > highThreshold || se > highThreshold) {
        } else {
          pDstRow[0] = 0;
        }
      }
    }
  }

  imshow("Show_Image", dstImage);
}
void addNoise(Mat& srcImage) {
  Mat noise = Mat(srcImage.size(), srcImage.type());
  randn(noise, 0, 20);
  srcImage = srcImage + noise;
  imshow("Show_Image", srcImage);
}

void houghTransform(Mat& dstImage) {
  Mat hough, cdst;
  hough = getGrayScaleImage(dstImage);
  // Copy edges to the images that will display the results in BGR
  cvtColor(hough, cdst, COLOR_GRAY2BGR);
  // Standard Hough Line Transform
  vector<Vec2f> lines;  // will hold the results of the detection
  HoughLines(cdst, lines, 1, CV_PI / 180, 150, 0,
             0);  // runs the actual detection
  // Draw the lines
  for (size_t i = 0; i < lines.size(); i++) {
    float rho = lines[i][0], theta = lines[i][1];
    Point pt1, pt2;
    double a = cos(theta), b = sin(theta);
    double x0 = a * rho, y0 = b * rho;
    pt1.x = cvRound(x0 + 1000 * (-b));
    pt1.y = cvRound(y0 + 1000 * (a));
    pt2.x = cvRound(x0 - 1000 * (-b));
    pt2.y = cvRound(y0 - 1000 * (a));
    line(cdst, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
  }

  imshow("Show_Image", cdst);
}