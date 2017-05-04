//
//  colorcode.cpp
//  opencv
//
//  Created by yu zhang on 2017/1/22.
//  Copyright © 2017年 yu zhang. All rights reserved.
//

#include "colorcode.hpp"
#include <stdlib.h>
#include <math.h>
#include <iostream>


typedef unsigned char uchar;

int ncols = 0;
#define MAXCOLS 60
static int colorwheel[MAXCOLS][3];

#ifndef M_PI
#define M_PI        3.14159265358979323846f
#endif


void setcols(int r, int g, int b, int k)
{
    colorwheel[k][0] = r;
    colorwheel[k][1] = g;
    colorwheel[k][2] = b;
}

void makecolorwheel()
{
    // relative lengths of color transitions:
    // these are chosen based on perceptual similarity
    // (e.g. one can distinguish more shades between red and yellow
    //  than between yellow and green)
    int RY = 15;
    int YG = 6;
    int GC = 4;
    int CB = 11;
    int BM = 13;
    int MR = 6;
    ncols = RY + YG + GC + CB + BM + MR;
    printf("ncols = %d\n", ncols);
    if (ncols > MAXCOLS)
        exit(1);
    int i;
    int k = 0;
    for (i = 0; i < RY; i++) setcols(255,      255*i/RY,     0,        k++);
    for (i = 0; i < YG; i++) setcols(255-255*i/YG, 255,      0,        k++);
    for (i = 0; i < GC; i++) setcols(0,        255,      255*i/GC,     k++);
    for (i = 0; i < CB; i++) setcols(0,        255-255*i/CB, 255,          k++);
    for (i = 0; i < BM; i++) setcols(255*i/BM,     0,        255,          k++);
    for (i = 0; i < MR; i++) setcols(255,      0,        255-255*i/MR, k++);
}

void computeColor(Vec2f& flow, Vec3b& pix, float max_rad)
{
    int RY = 15;
    int YG = 6;
    int GC = 4;
    int CB = 11;
    int BM = 13;
    int MR = 6;
    ncols = RY + YG + GC + CB + BM + MR;
    
    float fx = flow.val[0];
    float fy = flow.val[1];
    
    fx /= max_rad;
    fy /= max_rad;
    
    float rad = sqrtf(fx * fx + fy * fy);
    rad = round(rad);
    float a = atan2(-fy, -fx) / M_PI;
    float fk = (a + 1.0f) / 2.0f * (ncols-1);
    int k0 = (int)fk;
    int k1 = (k0 + 1) % ncols;
    float f = fk - k0;
    //f = 0; // uncomment to see original color wheel
    for (int b = 0; b < 3; b++) {
        float col0 = colorwheel[k0][b] / 255.0f;
        float col1 = colorwheel[k1][b] / 255.0f;
        float col = (1 - f) * col0 + f * col1;
        if (rad <= 1)
            col = 1 - rad * (1 - col); // increase saturation with radius
        else
            col *= .75; // out of range
        pix.val[2 - b] = (int)(255.0 * col);
        // pix.val[b] = (int)(255.0 * col);
    }
}

void computeColor(float flowx, float flowy, Vec3b& pix, float max_rad)
{
    int RY = 15;
    int YG = 6;
    int GC = 4;
    int CB = 11;
    int BM = 13;
    int MR = 6;
    ncols = RY + YG + GC + CB + BM + MR;
    
    float fx = flowx;
    float fy = flowy;
    
    fx /= max_rad;
    fy /= max_rad;
    
    float rad = sqrt(fx * fx + fy * fy);
    float a = atan2(-fy, -fx) / M_PI;
    float fk = (a + 1.0f) / 2.0f * (ncols-1);
    int k0 = (int)fk;
    int k1 = (k0 + 1) % ncols;
    float f = fk - k0;
    //f = 0; // uncomment to see original color wheel
    for (int b = 0; b < 3; b++) {
        float col0 = colorwheel[k0][b] / 255.0f;
        float col1 = colorwheel[k1][b] / 255.0f;
        float col = (1 - f) * col0 + f * col1;
        if (rad <= 1)
            col = 1 - rad * (1 - col); // increase saturation with radius
        else
            col *= .75; // out of range
        pix.val[2 - b] = (int)(255.0 * col);
    }
}

void computeColorMat(Mat& flowMat, Mat& imgMat, float max_disp_x, float max_disp_y) {
    
    float max_disp_rad = sqrt(max_disp_x*max_disp_x + max_disp_y*max_disp_y);
    
    for (int i = 0; i < flowMat.rows; ++i)
    {
        for (int j = 0; j < flowMat.cols; ++j)
        {
            computeColor(flowMat.at<Vec2f>(i,j), imgMat.at<Vec3b>(i,j), max_disp_rad);
        }
    }
}

void computeColorMat(Mat& flowMatX, Mat& flowMatY, Mat& imgMat, float max_disp_x, float max_disp_y) {
    float max_disp_rad = sqrt(max_disp_x*max_disp_x + max_disp_y*max_disp_y);
    for (int i = 0; i < flowMatX.rows; ++i)
    {
        for (int j = 0; j < flowMatX.cols; ++j)
        {
            computeColor(flowMatX.at<float>(i,j), flowMatY.at<float>(i,j), imgMat.at<Vec3b>(i,j), max_disp_rad);
        }
    }
}
