//
//  colorcode.hpp
//  opencv
//
//  Created by yu zhang on 2017/1/22.
//  Copyright © 2017年 yu zhang. All rights reserved.
//

#ifndef colorcode_hpp
#define colorcode_hpp

#include <stdio.h>
#include <vector>
#include <string>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

void makecolorwheel();

void computeColor(Vec2f& flow, Vec3b& pix);

void computeColor(float flowx, float flowy, Vec3b& pix, float);

void computeColorMat(Mat& flowMat, Mat& imgMat, float max_disp_x, float max_disp_y);

void computeColorMat(Mat& flowMatX, Mat& flowMatY, Mat& imgMat, float, float);


#endif /* colorcode_hpp */
