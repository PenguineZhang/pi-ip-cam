cmake_minimum_required(VERSION 3.0.0)
project(opencvtest)

set(CMAKE_CXX_STANDARD 17)

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_subdirectory(cpp-mjpeg-streamer)

add_executable(opencvtest main.cpp
						  motion_detector.cpp
						  motion_detector.hpp)

target_link_libraries( opencvtest ${OpenCV_LIBS}
								  nadjieb_mjpeg_streamer::nadjieb_mjpeg_streamer)
