//#include <slamBase.h>
//#include <cxeigen.hpp>
//#include <pcl/common/transforms.h>
#include "../include/slamBase.h"

namespace SLAMBase
{
	void showAndSaveImg(string imgName,cv::Mat& img)
	{
		cv::imshow(imgName.c_str(),img);
		cv::waitKey(0);
		cv::imwrite((dataPath+imgName+".png").c_str(),img);
	}

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
				p.x = ( c-cameraK.cx ) * p.z / cameraK.fx;
				p.y = ( r-cameraK.cy ) * p.z / cameraK.fy;

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

	cv::Point3f point2dTo3d( cv::Point3f& point,const CameraIntrinsicParameters& cameraK)
	{
		cv::Point3f p;
		p.z = double( point.z );
		p.x = ( point.x - cameraK.cx ) * p.z / cameraK.fx;
		p.y = ( point.y - cameraK.cy ) * p.z / cameraK.fy;
		return p;
	}

	void computeKeyPointsAndDesp(Frame& frame,const string& detector,const string& descriptor)
	{
		const cv::Mat& rgb = frame.rgb;
		const cv::Mat& depth = frame.depth;
        vector< cv::KeyPoint >& kp = frame.kp;
        cv::Mat& desp = frame.desp;
		cv::Ptr<cv::FeatureDetector> featureDetector = cv::FeatureDetector::create(detector);
		cv::Ptr<cv::DescriptorExtractor> descriptorExtractor = cv::DescriptorExtractor::create(descriptor);

        assert(featureDetector);
		assert(descriptorExtractor);

        featureDetector->detect(rgb, kp);
		descriptorExtractor->compute(rgb,kp,desp);

		return;
	}

	PnPResult estimateMotion(const Frame& frame1,const Frame& frame2,const CameraIntrinsicParameters& cameraK)
	{
		DC::Config configuration("../data/config.lua");

		vector< cv::DMatch > matches;
		cv::BFMatcher matcher;
		matcher.match( frame1.desp, frame2.desp, matches );
		cout << "matches number = " << matches.size() << endl;

		vector< cv::DMatch > refinedMatches;
        cv::Mat imgMatches;
		double minDis = 9999;
		double magnification = configuration.get<double>("FeatureDescriptorMatchingThresholdMagnification");
		for( size_t i=0; i<matches.size(); i++)
		{
			if( matches[i].distance < minDis )
				minDis = matches[i].distance;
		}
		for( size_t i=0;i <matches.size(); i++)
		{
			if( matches[i].distance < magnification*minDis )
				refinedMatches.push_back(matches[i]);
		}
		cout << "min dis = " << minDis << endl;
		cout << "good matches size = " << refinedMatches.size() << endl;
		cv::drawMatches( frame1.rgb, frame1.kp, frame2.rgb, frame2.kp, refinedMatches, imgMatches);
		showAndSaveImg("refiendMatches",imgMatches);

		vector< cv::Point3f > objPoints;
		vector< cv::Point2f > imgPoints;

		for(size_t i=0; i<refinedMatches.size(); i++)
		{
			cv::Point3f pt;
			pt.x = frame1.kp[refinedMatches[i].queryIdx].pt.x;
			pt.y = frame1.kp[refinedMatches[i].queryIdx].pt.y;
			pt.z = frame1.depth.ptr<ushort>(int(pt.y))[int(pt.x)];
            if( pt.z == 0 ) continue;
			//depth should be interpolated rather than get, should be modified

        	objPoints.push_back( point2dTo3d( pt , cameraK ) );
			imgPoints.push_back( frame2.kp[refinedMatches[i].trainIdx].pt );
		}

		PnPResult estimationResult;

        cv::Mat inliers;
        cv::solvePnPRansac( objPoints,
							imgPoints,
							*(cameraK.getCameraMatrix()),
							cv::Mat() ,
							estimationResult.rvec,
							estimationResult.tvec,
							configuration.get<bool>("PnPRansacUseExtrinsicGuess"),
							configuration.get<int>("PnPRansacIterationCount"),
							configuration.get<float>("PnPRansacReprojectionError"),
							configuration.get<int>("PnPRansacMinInliersCount"),
                            inliers
		);
		estimationResult.inliers = inliers.rows;
		estimationResult.tvec = estimationResult.tvec / cameraK.scale;
		vector< cv::DMatch > matchesShow;
		for( size_t i=0; i<inliers.rows; i++)
		{
			matchesShow.push_back( refinedMatches[inliers.ptr<int>(i)[0]] );
		}
		cv::drawMatches( frame1.rgb, frame1.kp, frame2.rgb, frame2.kp, matchesShow, imgMatches );
		showAndSaveImg("inlier matches",imgMatches);

		return estimationResult;
	}

	Eigen::Isometry3d cvMat2Eigen(
			const cv::Mat& rvec,
			const cv::Mat& tvec
	)
	{
		cv::Mat R;
		cv::Rodrigues(rvec, R);
		Eigen::Matrix3d r;
		cv::cv2eigen(R,r);

		Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
		Eigen::AngleAxisd angle(r);
		T = angle;
		T(0,3) = tvec.at<double>(0,0);
		T(1,3) = tvec.at<double>(0,1);
		T(2,3) = tvec.at<double>(0,2);

		return T;
	}
	PointCloud::Ptr jointPointCloud(PointCloud::Ptr cloud1,
									PointCloud::Ptr cloud2,
									Eigen::Isometry3d T
//									const CameraIntrinsicParameters& cameraK
	)
	{
		PointCloud::Ptr newCloud(new PointCloud());
		pcl::transformPointCloud( *cloud1, *newCloud, T.matrix());
		*newCloud += *cloud2;

        return newCloud;
//		static pcl::VoxelGrid<PointT> voxel;
//		static DC::Config configuration("../data/config.lua");
//		double gridSize = configuration.get<double>("VoxelGridSize");
//
//		voxel.setLeafSize(gridSize,gridSize,gridSize);
//		voxel.setInputCloud(newCloud);
//		PointCloud::Ptr filteredCloud( new PointCloud() );
//		voxel.filter(*filteredCloud);
//		return filteredCloud;
	}

}
