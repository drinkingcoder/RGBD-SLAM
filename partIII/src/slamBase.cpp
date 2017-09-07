#include "slamBase.h"

namespace SLAMBase
{
	PointCloud::Ptr image2PointCloud(cv::Mat& rgb,cv::Mat& depth,CameraIntrinsicParameters& cameraK)
	{
		PointCloud::Ptr cloud( new PointCloud );

		for(int r=0; r<depth.rows; r++)
			for(int c=0; c<depth.cols; c++)
			{
				ushort d = depth.ptr<ushort>(r)[c];

				if( d==0 ) continue;

				PointT p;

				p.z = double(d) / cameraK.scale;
				p.x = ( r-cameraK.cx ) * p.z / cameraK.fx;
				p.y = ( c-cameraK.cy ) * p.z / cameraK.fy;

				p.b = rgb.ptr<uchar>(r)[c*3];
				p.g = rgb.ptr<uchar>(r)[c*3+1];
				p.r = rgb.ptr<uchar>(r)[c*3+2];

				cloud->points.push_back(p);
			}

		cloud->height = 1;
		cloud->width = cloud->points.size();
		cloud->is_dense = false;

		return cloud;
	}

	cv::Point3f point2dTo3d( cv::Point3f& point, CameraIntrinsicParameters& cameraK)
	{
		cv::Point3f p;
		p.z = double( point.z );
		p.x = ( point.x - camera.cx ) * p.z / camera.fx;
		p.y = ( point.y - camera.cy ) * p.z / camera.fy;
		return p;
	}
};
