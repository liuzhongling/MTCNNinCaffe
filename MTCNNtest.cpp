// caffe
#include <caffe/caffe.hpp>
#include <caffe/layers/memory_data_layer.hpp>

// c++
#include <string>
#include <vector>
#include <fstream>
// opencv
#include <opencv2/opencv.hpp>
// boost
#include "boost/make_shared.hpp"
#include "mtcnn.h"

using namespace caffe;

int main(int argc, char **argv)
{
	if (argc != 3){
		std::cout << "MTMain.bin [model dir] [imagePath]" << std::endl;
		return 0;
	}
	::google::InitGoogleLogging(argv[0]);
	double threshold[3] = { 0.6, 0.7, 0.7 };
	double factor = 0.709;
	int minSize = 40;
	std::string proto_model_dir = argv[1];
	MTCNN detector(proto_model_dir);

	std::string imageName = argv[2];
	cv::Mat image = cv::imread(imageName);
	std::vector<FaceInfo> faceInfo;
	clock_t t1 = clock();
	std::cout << "Detect " << image.rows << "X" << image.cols;
	detector.Detect(image, faceInfo, minSize, threshold, factor);
#ifdef CPU_ONLY
	std::cout << " Time Using CPU: " << (clock() - t1)*1.0 / 1000 << std::endl;
#else
	std::cout << " Time Using GPU-CUDNN: " << (clock() - t1)*1.0 / 1000 << std::endl;
#endif
	for (int i = 0; i<faceInfo.size(); i++){
		float x = faceInfo[i].bbox.x1;
		float y = faceInfo[i].bbox.y1;
		float h = faceInfo[i].bbox.x2 - faceInfo[i].bbox.x1 + 1;
		float w = faceInfo[i].bbox.y2 - faceInfo[i].bbox.y1 + 1;
		cv::rectangle(image, cv::Rect(y, x, w, h), cv::Scalar(255, 0, 0), 2);
	}
	for (int i = 0; i<faceInfo.size(); i++){
		FacePts facePts = faceInfo[i].facePts;
		for (int j = 0; j<5; j++)
			cv::circle(image, cv::Point(facePts.y[j], facePts.x[j]), 1, cv::Scalar(255, 255, 0), 2);
	}
	cv::imshow("a", image);
	cv::waitKey(0);

	return 1;
}