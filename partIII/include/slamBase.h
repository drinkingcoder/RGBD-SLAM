#pragma once

#include<fstream>
#include<vector>

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>

#include<pcl/io/pcd_io.h>
#include<pcl/point_types.h>

using namespace std;

namespace SLAMBase
{

	typedef pcl::PointXYZRGBA PointT;
	typedef pcl::PointCloud<PointT> PointCloud;

	struct CameraIntrinsicParameters
	{
		double cx,cy,fx,fy,scale;
		CameraIntrinsicParameters(double cx,double cy,double fx,double fy,double scale):cx(cx),cy(cy),fx(fx),fy(fy),scale(scale)
		{}
	};

	PointCloud::Ptr image2PointCloud(cv::Mat& rgb, cv::Mat& depth, CameraIntrinsicParameters& cameraK);

	cv::Point3f point2dTo3d(cv::Point3f& point,const CameraIntrinsicParameters& cameraK);
};
