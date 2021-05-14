#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>  
using namespace std;
using namespace cv;
//绘制坐标轴
void myDrawFrameAxes(InputOutputArray image, InputArray cameraMatrix, InputArray distCoeffs,
    InputArray rvec, InputArray tvec, float length, int thickness);
//绘制边框以及id
void myDrawDetectedMarkers(InputOutputArray _image, InputArrayOfArrays _corners,
    InputArray _ids, Scalar borderColor);
//生成实际四个定点的坐标
void myGetSingleMarkerObjectPoints(float markerLength, OutputArray _objPoints);
//解PnP获得旋转向量和平移向量
void myEstimatePoseSingleMarkers(InputArrayOfArrays _corners, float markerLength,
    InputArray _cameraMatrix, InputArray _distCoeffs,
    OutputArray _rvecs, OutputArray _tvecs);
