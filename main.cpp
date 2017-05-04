//
//  main.cpp
//  OpticalFlowTools
//
//  Created by yu zhang on 2017/5/4.
//  Copyright © 2017年 yu zhang. All rights reserved.
//

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "flowIO.h"
#include "colorcode.hpp"

using namespace std;
using namespace cv;

int main(int argc, const char * argv[]) {
    // read data from .flo
    string filePath = "/Users/yuzhang/Documents/temp/flowDebugData/temp1/out.flo";
    Mat flowX, flowY;
    ReadFlowFile(filePath, flowX, flowY);
    
    // print data to color image
    Mat cMat(flowX.size(), CV_8UC3);
    float max_disp_x = 1;
    float max_disp_y = 1;
    makecolorwheel();
    
    computeColorMat(flowX, flowY, cMat, max_disp_x, max_disp_y);
    
    string tmpName = "/Users/yuzhang/Documents/temp/flowDebugData/temp1/flowNet.png";
    imwrite(tmpName, cMat);
    
    return 0;
}
