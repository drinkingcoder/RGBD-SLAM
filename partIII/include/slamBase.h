#pragma once

#include<fstream>
#include<vector>

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/nonfree/nonfree.hpp>
#include<opencv2/calib3d/calib3d.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>
//#include <pcl/filters/voxel_grid.h>

#include <opencv2/core/eigen.hpp>

#include "Config.hpp"

using namespace std;

namespace SLAMBase
{

	const string dataPath = "../data/";
	typedef pcl::PointXYZRGBA PointT;
	typedef pcl::PointCloud<PointT> PointCloud;

	struct CameraIntrinsicParameters
	{
		double cx,cy,fx,fy,scale;
        double camera_matrix_data[3][3];
		std::shared_ptr<cv::Mat> m_cameraMatrix;

		void initCameraMatrix() {
			m_cameraMatrix = std::make_shared<cv::Mat>(3,3,CV_64F,camera_matrix_data);
			std::cout << " Camera Matrix= " << std::endl
											<< *m_cameraMatrix << std::endl;
		}

		std::shared_ptr<cv::Mat> getCameraMatrix ()
		{
			return m_cameraMatrix;
		}

		std::shared_ptr<cv::Mat> getCameraMatrix() const
		{
			return m_cameraMatrix;
		}

		CameraIntrinsicParameters(double cx,double cy,double fx,double fy,double scale):
				cx(cx),
				cy(cy),
				fx(fx),
				fy(fy),
				scale(scale)
		{
			memset(camera_matrix_data,0,sizeof(camera_matrix_data));
			camera_matrix_data[0][0] = fx;
			camera_matrix_data[0][2] = cx;
			camera_matrix_data[1][1] = fy;
			camera_matrix_data[1][2] = cy;
			camera_matrix_data[2][2] = 1;
			initCameraMatrix();
		}

	};

	PointCloud::Ptr image2PointCloud(cv::Mat& rgb, cv::Mat& depth, CameraIntrinsicParameters& cameraK);

	cv::Point3f point2dTo3d(cv::Point3f& point,const CameraIntrinsicParameters& cameraK);

	struct Frame
	{
		cv::Mat rgb,depth;
		cv::Mat desp;
		vector< cv::KeyPoint > kp;
	};

	struct PnPResult
	{
		cv::Mat rvec,tvec;
		int inliers;
	};

	void computeKeyPointsAndDesp(Frame& frame,
								 const string& detector,
								 const string& descriptor
	);
	PnPResult estimateMotion(const Frame& frame1,
							 const Frame& frame2,
							 const CameraIntrinsicParameters& cameraK
	);
	Eigen::Isometry3d cvMat2Eigen(
			const cv::Mat& rvec,
			const cv::Mat& tvec
	);
	PointCloud::Ptr jointPointCloud(PointCloud::Ptr cloud1,
									PointCloud::Ptr cloud2,
									Eigen::Isometry3d T
//									const CameraIntrinsicParameters& camera
	);
};
