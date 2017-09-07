#include<iostream>
#include<string>
#include "../include/slamBase.h"

using namespace std;
using namespace SLAMBase;

#include<opencv2/features2d/features2d.hpp>
#include<opencv2/nonfree/nonfree.hpp>
#include<opencv2/calib3d/calib3d.hpp>

const string dataPath = "../data/";

void inputData(cv::Mat& rgb1, cv::Mat& depth1, cv::Mat& rgb2, cv::Mat& depth2)
{
	rgb1 = cv::imread("../data/rgb1.png");
	rgb2 = cv::imread("../data/rgb2.png");

	depth1 = cv::imread("../data/depth1.png");
	depth2 = cv::imread("../data/depth2.png");
}

void showAndSaveImg(string imgName,cv::Mat& img)
{
	cv::imshow(imgName.c_str(),img);
	cv::waitKey(0);
	cv::imwrite((dataPath+imgName).c_str(),img);
}


int main( int argc, char** argv )
{
	cv::Mat rgb1,rgb2,depth1,depth2;
	cv::Ptr<cv::FeatureDetector> featureDetector;
	cv::Ptr<cv::DescriptorExtractor> descriptorExtractor;
	vector< cv::KeyPoint > kp1,kp2;
	cv::Mat imgShow;
	cv::initModule_nonfree();

	//read in image data
	InputData( rgb1, depth1, rgb2, depth2 );

	//create feature detecotor & descriptor extractor
	featureDetector = cv::FeatureDetector::create("GridSIFT");
	descriptorExtractor = cv::DescriptorExtractor::creat("SIFT");

	//detect keypoint2
	featureDetector->detect(rgb1,kp1);
	featureDetector->detect(rgb2,kp2);
	cout << "Key points of two images: " << kp1.size() << ", "
		<< kp2.size() << endl;

	//show keypoints on the original image
	cv::drawKeypoints( rgb1, kp1, imgShow, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	showAndSaveImg("keypoints",imgShow);

	//extract descriptor
	cv::Mat desp1, desp2;
	descriptorExtractor->compute( rgb1, kp1, desp1);
	descriptorExtractor->compute( rgb2, kp2, desp2);

	//rough feature matching
	vector< cv::DMatch > matches;
	cv::FlannBasedMatcher matcher;
	matcher.match( desp1. deps2, matches );
	cout << "matches number = " << matches.size() << endl;

	//show matches
	cv::Mat imgMatches;
	cv::drawMatches( rgb1, kp1, rgb2, kp2, matches, imgMatches);
	showAndSaveImg("matches",imgMatches);

	//bad matches resigning
	vector< cv::DMatch > refinedMatches;
	double minDis = 9999;
	for( size_t i=0; i<matches.size(); i++)
	{
		if( matches[i].distance < minDis )
		   	minDis = matches[i].distance;
	}
	for( size_t i=0;i <matches.size(); i++)
	{
		if( matches[i].distance < 4*minDis ) 
			refinedMatches.push_back(matches[i]);
	}
	cout << "good matches size = " << refinedMatches.size() << endl;
	cv::drawMatches(rgb1, kp1, rgb2, kp2, refinedMatches, imgMatches);
	showAndSaveImg("refiendMatches.png",imgMatches);

	vector< cv::Point3f > objPoints;
	vector< cv::Point2f > imgPoints;
}
