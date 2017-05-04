#ifndef _FLOW_IO_H_
#define _FLOW_IO_H_

#include <opencv2/core.hpp>
#include <iostream>


using namespace cv;
using namespace std;


void WriteFlowFile(Mat, Mat, string filename);

void ReadFlowFile(string, Mat&, Mat&);


#endif
