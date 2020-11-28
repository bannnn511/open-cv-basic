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

  int highThreshold = 91, lowThreshold = 31;
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
        // edge directions are classified into four group of angles 0, 45, 90,
        // 135 degrees
        float angle = round((atan(gx / gy)) / 45) * 45;

        // if (g > 80) {
        //   pDstRow[0] = 255;
        // }
        pDstRow[0] = g;
        pDirRow[0] = angle;
      }
    }
  }

  // restart position
  pSrcData = (uchar*)srcImage.data + yStart * widthStep + xStart * nChannels;
  pDstData = (uchar*)dstImage.data + yStart * widthStep + xStart * nChannels;
  pDirData =
      (uchar*)gradientDirection.data + yStart * widthStep + xStart * nChannels;

  // Non-maximum Suppression + Hysteresis Thresholding
  for (int y = yStart; y <= yEnd; y++, pSrcData += widthStep,
           pDstData += widthStep, pDirData += widthStep) {
    const uchar* pSrcRow = pSrcData;
    uchar* pDstRow = pDstData;
    uchar* pDirRow = pDirData;
    for (int x = xStart; x <= xEnd; x++, pSrcRow += nChannels,
             pDstRow += nChannels, pDirRow += nChannels) {
      uchar* pDstRowN = NULL;
      uchar* pDstRowS = NULL;
      if (pDirRow[0] == 0) {
        pDstRowN = pDstRow - 1;
        pDstRowS = pDstRow + 1;
      } else if (pDirRow[0] == 45) {
        pDstRowN = pDstRow - widthStep + 1;
        pDstRowS = pDstRow + widthStep - 1;
      } else if (pDstRow[0] == 90) {
        pDstRowN = pDstRow - widthStep;
        pDstRowS = pDstRow + widthStep;
      } else if (pDstRow[0] == 135) {
        pDstRowN = pDstRow - widthStep - 1;
        pDstRowS = pDstRow + widthStep + 1;
      }

      if (pDstRowN != NULL && pDstRowS != NULL) {
        if (pDstRow[0] < pDstRowN[0] || pDstRow[0] < pDstRowS[0]) {
          pDstRow[0] = 0;
        }
        if (pDstRow[0] < lowThreshold) {
          pDstRow[0] = 0;
        }
        if (pDstRow[0] > lowThreshold && pDstRow[0] < highThreshold) {
          uchar* nw = pDstRow - widthStep - 1;
          uchar* n = pDstRow - widthStep;
          uchar* ne = pDstRow - widthStep + 1;
          uchar* w = pDstRow - 1;
          uchar* e = pDstRow + 1;
          uchar* sw = pDstRow + widthStep - 1;
          uchar* s = pDstRow + widthStep;
          uchar* se = pDstRow + widthStep + 1;
          if (nw[0] > highThreshold || n[0] > highThreshold ||
              ne[0] > highThreshold || w[0] > highThreshold ||
              e[0] > highThreshold || sw[0] > highThreshold ||
              s[0] > highThreshold || se[0] > highThreshold) {
          } else {
            pDstRow[0] = 0;
          }
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
