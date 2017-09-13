#include <iostream>

#include "../include/slamBase.h"

#include <opencv2/core/eigen.hpp>
//
//#include <pcl/common/transforms.h>
//#include <pcl/visualization/cloud_viewer.h>

using namespace SLAMBase;

void InputData(Frame& frame1,Frame& frame2,DC::Config& configuration)
{
    frame1.depth = cv::imread(configuration.get<string>("FrameDepthName1"), -1);
    frame1.rgb = cv::imread(configuration.get<string>("FrameRGBName1"));
    frame2.depth = cv::imread(configuration.get<string>("FrameDepthName2"), -1);
    frame2.rgb = cv::imread(configuration.get<string>("FrameRGBName2"));
}

int main(int argc, char** argv)
{
    DC::Config configuration("../data/config.lua");
    Frame frame1,frame2;
    InputData( frame1 , frame2 , configuration );

    string detector = configuration.get<string>("DetectorName");
    string descriptor = configuration.get<string>("DescriptorName");
    computeKeyPointsAndDesp(frame1,detector,descriptor);
    computeKeyPointsAndDesp(frame2,detector,descriptor);

    double fx = configuration.get<double>("CameraIntrinsicsfx");
    double fy = configuration.get<double>("CameraIntrinsicsfy");
    double cx = configuration.get<double>("CameraIntrinsicscx");
    double cy = configuration.get<double>("CameraIntrinsicscy");
    double scale = configuration.get<double>("CameraIntrinsicsscale");
    CameraIntrinsicParameters cameraK( cx, cy, fx, fy, scale);

    PnPResult result = estimateMotion( frame1, frame2, cameraK);
    std::cout << "inlier size = " << result.inliers << std::endl;

    PointCloud::Ptr cloud1 = image2PointCloud( frame1.rgb, frame1.depth, cameraK);
    pcl::io::savePCDFile("../data/frame1.pcd", *cloud1);
    PointCloud::Ptr cloud2 = image2PointCloud( frame2.rgb, frame2.depth, cameraK);

    Eigen::Isometry3d T = cvMat2Eigen(result.rvec,result.tvec);

    PointCloud::Ptr outputCloud = jointPointCloud(cloud1,cloud2,T);
    pcl::io::savePCDFile("../data/result.pcd", *outputCloud);
    std::cout << "Result saved" <<std::endl;

    return 0;
}
