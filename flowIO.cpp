#include "flowIO.h"

// first four bytes, should be the same in little endian
#define TAG_FLOAT 202021.25  // check for this when READING the file
#define TAG_STRING "PIEH"    // use this when WRITING the file

using namespace std;
using namespace cv;

void ReadFlowFile(string filename, Mat& flowX, Mat& flowY) {
    if(filename == ""){
        cout << "empty filename" << endl;
        return;
    }
    
    FILE *stream = fopen(filename.c_str(), "rb");
    if(stream == 0) {
        cout << "file open error" << endl;
        return;
    }
    
    int width, height;
    float tag;
    
    // read file header
    if ((int)fread(&tag, sizeof(float), 1, stream) != 1 ||
        (int)fread(&width, sizeof(int), 1, stream) != 1 ||
        (int)fread(&height, sizeof(int), 1, stream) != 1) {
        cout << "read file head error" << endl;
        return;
    }
    
    if (tag != TAG_FLOAT) {
        cout << "wrong tag" << endl;
        return;
    }
    if (width < 1 || width > 99999) {
        cout << "wrong width value" << endl;
        return;
    }
    if (height < 1 || height > 99999) {
        cout << "wrong height value" << endl;
    }
    
    flowX.create(Size(width, height), CV_32F);
    flowY.create(Size(width, height), CV_32F);
    
    // read file body
    float ptr[] = {0.0f, 0.0f};
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            fread(ptr, sizeof(float), 2, stream);
            flowX.at<float>(h,w) = ptr[0];
            flowY.at<float>(h,w) = ptr[1];
        }
    }
}


void WriteFlowFile(Mat flowX, Mat flowY, string filename) {

    int width = flowX.cols, height = flowX.rows;

    cout << "width: " << width << ", height: " << height << endl;;



    FILE *stream = fopen(filename.c_str(), "wb");
    if (stream == 0) {
    	cout << "WriteFlowFile: could not open " << filename << endl;
    	return;
	}

	// write the header
	fprintf(stream, TAG_STRING);
	if ((int)fwrite(&width, sizeof(int), 1, stream) != 1 ||
		(int)fwrite(&height, sizeof(int), 1, stream) != 1) {
		cout << "WriteFlowFile: problem writing header " << filename << endl;
    	return;
	}

   for (int y = 0; y < height; ++y) {
   		for (int x = 0; x < width; ++x) {
   			float  ptr[] = {flowX.at<float>(y,x), flowY.at<float>(y,x)};
			if ((int)fwrite(ptr, sizeof(float), 2, stream) != 2) {
				cout << "WriteFlowFile: problem writing data " << filename << endl;
	    		return;
	    	}
		}
   }

    fclose(stream);
}
