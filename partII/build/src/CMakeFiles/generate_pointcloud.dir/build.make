# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/drinkingcoder/Documents/rgbd-slam/partII

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/drinkingcoder/Documents/rgbd-slam/partII/build

# Include any dependencies generated for this target.
include src/CMakeFiles/generate_pointcloud.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/generate_pointcloud.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/generate_pointcloud.dir/flags.make

src/CMakeFiles/generate_pointcloud.dir/generatePointCloud.cpp.o: src/CMakeFiles/generate_pointcloud.dir/flags.make
src/CMakeFiles/generate_pointcloud.dir/generatePointCloud.cpp.o: ../src/generatePointCloud.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/drinkingcoder/Documents/rgbd-slam/partII/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object src/CMakeFiles/generate_pointcloud.dir/generatePointCloud.cpp.o"
	cd /home/drinkingcoder/Documents/rgbd-slam/partII/build/src && g++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/generate_pointcloud.dir/generatePointCloud.cpp.o -c /home/drinkingcoder/Documents/rgbd-slam/partII/src/generatePointCloud.cpp

src/CMakeFiles/generate_pointcloud.dir/generatePointCloud.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/generate_pointcloud.dir/generatePointCloud.cpp.i"
	cd /home/drinkingcoder/Documents/rgbd-slam/partII/build/src && g++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/drinkingcoder/Documents/rgbd-slam/partII/src/generatePointCloud.cpp > CMakeFiles/generate_pointcloud.dir/generatePointCloud.cpp.i

src/CMakeFiles/generate_pointcloud.dir/generatePointCloud.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/generate_pointcloud.dir/generatePointCloud.cpp.s"
	cd /home/drinkingcoder/Documents/rgbd-slam/partII/build/src && g++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/drinkingcoder/Documents/rgbd-slam/partII/src/generatePointCloud.cpp -o CMakeFiles/generate_pointcloud.dir/generatePointCloud.cpp.s

src/CMakeFiles/generate_pointcloud.dir/generatePointCloud.cpp.o.requires:
.PHONY : src/CMakeFiles/generate_pointcloud.dir/generatePointCloud.cpp.o.requires

src/CMakeFiles/generate_pointcloud.dir/generatePointCloud.cpp.o.provides: src/CMakeFiles/generate_pointcloud.dir/generatePointCloud.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/generate_pointcloud.dir/build.make src/CMakeFiles/generate_pointcloud.dir/generatePointCloud.cpp.o.provides.build
.PHONY : src/CMakeFiles/generate_pointcloud.dir/generatePointCloud.cpp.o.provides

src/CMakeFiles/generate_pointcloud.dir/generatePointCloud.cpp.o.provides.build: src/CMakeFiles/generate_pointcloud.dir/generatePointCloud.cpp.o

# Object files for target generate_pointcloud
generate_pointcloud_OBJECTS = \
"CMakeFiles/generate_pointcloud.dir/generatePointCloud.cpp.o"

# External object files for target generate_pointcloud
generate_pointcloud_EXTERNAL_OBJECTS =

../bin/generate_pointcloud: src/CMakeFiles/generate_pointcloud.dir/generatePointCloud.cpp.o
../bin/generate_pointcloud: src/CMakeFiles/generate_pointcloud.dir/build.make
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.2.4.8
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libopencv_video.so.2.4.8
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libopencv_ts.so.2.4.8
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.2.4.8
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.2.4.8
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.2.4.8
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libopencv_ocl.so.2.4.8
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.2.4.8
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.2.4.8
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libopencv_legacy.so.2.4.8
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.8
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.8
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libopencv_gpu.so.2.4.8
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.2.4.8
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.2.4.8
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.8
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libopencv_contrib.so.2.4.8
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.2.4.8
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libboost_system.so
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libboost_thread.so
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libpthread.so
../bin/generate_pointcloud: /usr/lib/libpcl_common.so
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
../bin/generate_pointcloud: /usr/lib/libpcl_kdtree.so
../bin/generate_pointcloud: /usr/lib/libpcl_octree.so
../bin/generate_pointcloud: /usr/lib/libpcl_search.so
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libqhull.so
../bin/generate_pointcloud: /usr/lib/libpcl_surface.so
../bin/generate_pointcloud: /usr/lib/libpcl_sample_consensus.so
../bin/generate_pointcloud: /usr/lib/libOpenNI.so
../bin/generate_pointcloud: /usr/lib/libOpenNI2.so
../bin/generate_pointcloud: /usr/lib/libvtkCommon.so.5.8.0
../bin/generate_pointcloud: /usr/lib/libvtkFiltering.so.5.8.0
../bin/generate_pointcloud: /usr/lib/libvtkImaging.so.5.8.0
../bin/generate_pointcloud: /usr/lib/libvtkGraphics.so.5.8.0
../bin/generate_pointcloud: /usr/lib/libvtkGenericFiltering.so.5.8.0
../bin/generate_pointcloud: /usr/lib/libvtkIO.so.5.8.0
../bin/generate_pointcloud: /usr/lib/libvtkRendering.so.5.8.0
../bin/generate_pointcloud: /usr/lib/libvtkVolumeRendering.so.5.8.0
../bin/generate_pointcloud: /usr/lib/libvtkHybrid.so.5.8.0
../bin/generate_pointcloud: /usr/lib/libvtkWidgets.so.5.8.0
../bin/generate_pointcloud: /usr/lib/libvtkParallel.so.5.8.0
../bin/generate_pointcloud: /usr/lib/libvtkInfovis.so.5.8.0
../bin/generate_pointcloud: /usr/lib/libvtkGeovis.so.5.8.0
../bin/generate_pointcloud: /usr/lib/libvtkViews.so.5.8.0
../bin/generate_pointcloud: /usr/lib/libvtkCharts.so.5.8.0
../bin/generate_pointcloud: /usr/lib/libpcl_io.so
../bin/generate_pointcloud: /usr/lib/libpcl_filters.so
../bin/generate_pointcloud: /usr/lib/libpcl_features.so
../bin/generate_pointcloud: /usr/lib/libpcl_keypoints.so
../bin/generate_pointcloud: /usr/lib/libpcl_registration.so
../bin/generate_pointcloud: /usr/lib/libpcl_segmentation.so
../bin/generate_pointcloud: /usr/lib/libpcl_recognition.so
../bin/generate_pointcloud: /usr/lib/libpcl_visualization.so
../bin/generate_pointcloud: /usr/lib/libpcl_people.so
../bin/generate_pointcloud: /usr/lib/libpcl_outofcore.so
../bin/generate_pointcloud: /usr/lib/libpcl_tracking.so
../bin/generate_pointcloud: /usr/lib/libpcl_apps.so
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libboost_system.so
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libboost_thread.so
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libpthread.so
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libqhull.so
../bin/generate_pointcloud: /usr/lib/libOpenNI.so
../bin/generate_pointcloud: /usr/lib/libOpenNI2.so
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
../bin/generate_pointcloud: /usr/lib/libvtkCommon.so.5.8.0
../bin/generate_pointcloud: /usr/lib/libvtkFiltering.so.5.8.0
../bin/generate_pointcloud: /usr/lib/libvtkImaging.so.5.8.0
../bin/generate_pointcloud: /usr/lib/libvtkGraphics.so.5.8.0
../bin/generate_pointcloud: /usr/lib/libvtkGenericFiltering.so.5.8.0
../bin/generate_pointcloud: /usr/lib/libvtkIO.so.5.8.0
../bin/generate_pointcloud: /usr/lib/libvtkRendering.so.5.8.0
../bin/generate_pointcloud: /usr/lib/libvtkVolumeRendering.so.5.8.0
../bin/generate_pointcloud: /usr/lib/libvtkHybrid.so.5.8.0
../bin/generate_pointcloud: /usr/lib/libvtkWidgets.so.5.8.0
../bin/generate_pointcloud: /usr/lib/libvtkParallel.so.5.8.0
../bin/generate_pointcloud: /usr/lib/libvtkInfovis.so.5.8.0
../bin/generate_pointcloud: /usr/lib/libvtkGeovis.so.5.8.0
../bin/generate_pointcloud: /usr/lib/libvtkViews.so.5.8.0
../bin/generate_pointcloud: /usr/lib/libvtkCharts.so.5.8.0
../bin/generate_pointcloud: /usr/lib/libpcl_common.so
../bin/generate_pointcloud: /usr/lib/libpcl_kdtree.so
../bin/generate_pointcloud: /usr/lib/libpcl_octree.so
../bin/generate_pointcloud: /usr/lib/libpcl_search.so
../bin/generate_pointcloud: /usr/lib/libpcl_surface.so
../bin/generate_pointcloud: /usr/lib/libpcl_sample_consensus.so
../bin/generate_pointcloud: /usr/lib/libpcl_io.so
../bin/generate_pointcloud: /usr/lib/libpcl_filters.so
../bin/generate_pointcloud: /usr/lib/libpcl_features.so
../bin/generate_pointcloud: /usr/lib/libpcl_keypoints.so
../bin/generate_pointcloud: /usr/lib/libpcl_registration.so
../bin/generate_pointcloud: /usr/lib/libpcl_segmentation.so
../bin/generate_pointcloud: /usr/lib/libpcl_recognition.so
../bin/generate_pointcloud: /usr/lib/libpcl_visualization.so
../bin/generate_pointcloud: /usr/lib/libpcl_people.so
../bin/generate_pointcloud: /usr/lib/libpcl_outofcore.so
../bin/generate_pointcloud: /usr/lib/libpcl_tracking.so
../bin/generate_pointcloud: /usr/lib/libpcl_apps.so
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.2.4.8
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libopencv_legacy.so.2.4.8
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libopencv_video.so.2.4.8
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.2.4.8
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.2.4.8
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.2.4.8
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.2.4.8
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.8
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.8
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.2.4.8
../bin/generate_pointcloud: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.8
../bin/generate_pointcloud: /usr/lib/libvtkViews.so.5.8.0
../bin/generate_pointcloud: /usr/lib/libvtkInfovis.so.5.8.0
../bin/generate_pointcloud: /usr/lib/libvtkWidgets.so.5.8.0
../bin/generate_pointcloud: /usr/lib/libvtkVolumeRendering.so.5.8.0
../bin/generate_pointcloud: /usr/lib/libvtkHybrid.so.5.8.0
../bin/generate_pointcloud: /usr/lib/libvtkParallel.so.5.8.0
../bin/generate_pointcloud: /usr/lib/libvtkRendering.so.5.8.0
../bin/generate_pointcloud: /usr/lib/libvtkImaging.so.5.8.0
../bin/generate_pointcloud: /usr/lib/libvtkGraphics.so.5.8.0
../bin/generate_pointcloud: /usr/lib/libvtkIO.so.5.8.0
../bin/generate_pointcloud: /usr/lib/libvtkFiltering.so.5.8.0
../bin/generate_pointcloud: /usr/lib/libvtkCommon.so.5.8.0
../bin/generate_pointcloud: /usr/lib/libvtksys.so.5.8.0
../bin/generate_pointcloud: src/CMakeFiles/generate_pointcloud.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable ../../bin/generate_pointcloud"
	cd /home/drinkingcoder/Documents/rgbd-slam/partII/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/generate_pointcloud.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/generate_pointcloud.dir/build: ../bin/generate_pointcloud
.PHONY : src/CMakeFiles/generate_pointcloud.dir/build

src/CMakeFiles/generate_pointcloud.dir/requires: src/CMakeFiles/generate_pointcloud.dir/generatePointCloud.cpp.o.requires
.PHONY : src/CMakeFiles/generate_pointcloud.dir/requires

src/CMakeFiles/generate_pointcloud.dir/clean:
	cd /home/drinkingcoder/Documents/rgbd-slam/partII/build/src && $(CMAKE_COMMAND) -P CMakeFiles/generate_pointcloud.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/generate_pointcloud.dir/clean

src/CMakeFiles/generate_pointcloud.dir/depend:
	cd /home/drinkingcoder/Documents/rgbd-slam/partII/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/drinkingcoder/Documents/rgbd-slam/partII /home/drinkingcoder/Documents/rgbd-slam/partII/src /home/drinkingcoder/Documents/rgbd-slam/partII/build /home/drinkingcoder/Documents/rgbd-slam/partII/build/src /home/drinkingcoder/Documents/rgbd-slam/partII/build/src/CMakeFiles/generate_pointcloud.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/generate_pointcloud.dir/depend

