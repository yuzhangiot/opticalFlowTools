#include "opencv2/highgui.hpp"
#include "opencv2/video.hpp"
#include "opencv2/optflow.hpp"
#include "opencv2/core/ocl.hpp"
#include <fstream>
#include <limits>
#include <iostream>

using namespace std;
using namespace cv;
using namespace optflow;

const String keys = "{help h usage ? |      | print this message   }"
        "{@image1        |      | image1               }"
        "{@image2        |      | image2               }"
        "{@algorithm     |      | [farneback, simpleflow, tvl1, deepflow, sparsetodenseflow, pcaflow, DISflow_ultrafast, DISflow_fast, DISflow_medium] }"
        "{@groundtruth   |      | path to the .flo file  (optional), Middlebury format }"
        "{m measure      |endpoint| error measure - [endpoint or angular] }"
        "{r region       |all   | region to compute stats about [all, discontinuities, untextured] }"
        "{d display      |      | display additional info images (pauses program execution) }"
        "{g gpu          |      | use OpenCL}"
        "{prior          |      | path to a prior file for PCAFlow}";

inline bool isFlowCorrect( const Point2f u )
{
    return !cvIsNaN(u.x) && !cvIsNaN(u.y) && (fabs(u.x) < 1e9) && (fabs(u.y) < 1e9);
}
inline bool isFlowCorrect( const Point3f u )
{
    return !cvIsNaN(u.x) && !cvIsNaN(u.y) && !cvIsNaN(u.z) && (fabs(u.x) < 1e9) && (fabs(u.y) < 1e9)
            && (fabs(u.z) < 1e9);
}
static Mat endpointError( const Mat_<Point2f>& flow1, const Mat_<Point2f>& flow2 )
{
    Mat result(flow1.size(), CV_32FC1);
    for ( int i = 0; i < flow1.rows; ++i )
    {
        for ( int j = 0; j < flow1.cols; ++j )
        {
            const Point2f u1 = flow1(i, j);
            const Point2f u2 = flow2(i, j);

            if ( isFlowCorrect(u1) && isFlowCorrect(u2) )
            {
                const Point2f diff = u1 - u2;
                result.at<float>(i, j) = sqrt((float)diff.ddot(diff)); //distance
            } else
                result.at<float>(i, j) = std::numeric_limits<float>::quiet_NaN();
        }
    }
    return result;
}
static Mat angularError( const Mat_<Point2f>& flow1, const Mat_<Point2f>& flow2 )
{
    Mat result(flow1.size(), CV_32FC1);

    for ( int i = 0; i < flow1.rows; ++i )
    {
        for ( int j = 0; j < flow1.cols; ++j )
        {
            const Point2f u1_2d = flow1(i, j);
            const Point2f u2_2d = flow2(i, j);
            const Point3f u1(u1_2d.x, u1_2d.y, 1);
            const Point3f u2(u2_2d.x, u2_2d.y, 1);

            if ( isFlowCorrect(u1) && isFlowCorrect(u2) )
                result.at<float>(i, j) = acos((float)(u1.ddot(u2) / norm(u1) * norm(u2)));
            else
                result.at<float>(i, j) = std::numeric_limits<float>::quiet_NaN();
        }
    }
    return result;
}
// what fraction of pixels have errors higher than given threshold?
static float stat_RX( Mat errors, float threshold, Mat mask )
{
    CV_Assert(errors.size() == mask.size());
    CV_Assert(mask.depth() == CV_8U);

    int count = 0, all = 0;
    for ( int i = 0; i < errors.rows; ++i )
    {
        for ( int j = 0; j < errors.cols; ++j )
        {
            if ( mask.at<char>(i, j) != 0 )
            {
                ++all;
                if ( errors.at<float>(i, j) > threshold )
                    ++count;
            }
        }
    }
    return (float)count / all;
}
static float stat_AX( Mat hist, int cutoff_count, float max_value )
{
    int counter = 0;
    int bin = 0;
    int bin_count = hist.rows;
    while ( bin < bin_count && counter < cutoff_count )
    {
        counter += (int) hist.at<float>(bin, 0);
        ++bin;
    }
    return (float) bin / bin_count * max_value;
}
static void calculateStats( Mat errors, Mat mask = Mat(), bool display_images = false )
{
    float R_thresholds[] = { 0.5f, 1.f, 2.f, 5.f, 10.f };
    float A_thresholds[] = { 0.5f, 0.75f, 0.95f };
    if ( mask.empty() )
        mask = Mat::ones(errors.size(), CV_8U);
    CV_Assert(errors.size() == mask.size());
    CV_Assert(mask.depth() == CV_8U);

    //mean and std computation
    Scalar s_mean, s_std;
    float mean, std;
    meanStdDev(errors, s_mean, s_std, mask);
    mean = (float)s_mean[0];
    std = (float)s_std[0];
    printf("Average: %.2f\nStandard deviation: %.2f\n", mean, std);

    //RX stats - displayed in percent
    float R;
    int R_thresholds_count = sizeof(R_thresholds) / sizeof(float);
    for ( int i = 0; i < R_thresholds_count; ++i )
    {
        R = stat_RX(errors, R_thresholds[i], mask);
        printf("R%.1f: %.2f%%\n", R_thresholds[i], R * 100);
    }


/*
    //AX stats
    double max_value;
    minMaxLoc(errors, NULL, &max_value, NULL, NULL, mask);

    Mat hist;
    const int n_images = 1;
    const int channels[] = { 0 };
    const int n_dimensions = 1;
    const int hist_bins[] = { 1024 };
    const float iranges[] = { 0, (float) max_value };
    const float* ranges[] = { iranges };
    const bool uniform = true;
    const bool accumulate = false;
    calcHist(&errors, n_images, channels, mask, hist, n_dimensions, hist_bins, ranges, uniform,
            accumulate);
    int all_pixels = countNonZero(mask);
    int cutoff_count;
    float A;
    int A_thresholds_count = sizeof(A_thresholds) / sizeof(float);
    for ( int i = 0; i < A_thresholds_count; ++i )
    {
        cutoff_count = (int) (floor(A_thresholds[i] * all_pixels + 0.5f));
        A = stat_AX(hist, cutoff_count, (float) max_value);
        printf("A%.2f: %.2f\n", A_thresholds[i], A);
    }
*/
}

static Mat flowToDisplay(const Mat flow)
{
    Mat flow_split[2];
    Mat magnitude, angle;
    Mat hsv_split[3], hsv, rgb;
    split(flow, flow_split);
    cartToPolar(flow_split[0], flow_split[1], magnitude, angle, true);
    normalize(magnitude, magnitude, 0, 1, NORM_MINMAX);
    hsv_split[0] = angle; // already in degrees - no normalization needed
    hsv_split[1] = Mat::ones(angle.size(), angle.type());
    hsv_split[2] = magnitude;
    merge(hsv_split, 3, hsv);
    cvtColor(hsv, rgb, COLOR_HSV2BGR);
    return rgb;
}

//int main( int argc, char** argv )
//{
//    CommandLineParser parser(argc, argv, keys);
//    parser.about("OpenCV optical flow evaluation app");
//    String myflo_path = parser.get<String>(0);
//    String groundtruth_path = parser.get<String>(1);
//    String error_measure = parser.get<String>("measure");
//
//    Mat_<Point2f> flow, ground_truth;
//    Mat computed_errors;
//
//    double startTick, time;
//    startTick = (double) getTickCount(); // measure time
//
//    time = ((double) getTickCount() - startTick) / getTickFrequency();
//    printf("\nTime [s]: %.3f\n", time);
//
//    if ( !groundtruth_path.empty() )
//    { // compare to ground truth
//        ground_truth = optflow::readOpticalFlow(groundtruth_path);
//        flow = optflow::readOpticalFlow(myflo_path);
//        if ( flow.size() != ground_truth.size() || flow.channels() != 2
//                || ground_truth.channels() != 2 )
//        {
//            printf("Dimension mismatch between the computed flow and the provided ground truth\n");
//            return -1;
//        }
//        if ( error_measure == "endpoint" ) {
//            cout << "flow size is: " << flow.size() << endl;
//            cout << "groundtruth size is: " << ground_truth.size() << endl; 
//            computed_errors = endpointError(flow, ground_truth);
//        }
//       // else if ( error_measure == "angular" )
//         //   computed_errors = angularError(flow, ground_truth);
//        else
//        {
//            printf("Invalid error measure! Available options: endpoint, angular\n");
//            return -1;
//        }
//
//        Mat mask;
//        mask = Mat::ones(ground_truth.size(), CV_8U) * 255;
//
//        printf("Using %s error measure\n", error_measure.c_str());
//        calculateStats(computed_errors, mask, false);
//
//    }
//
//    return 0;
//
//}
