#include <nadjieb/mjpeg_streamer.hpp>

#include <opencv2/opencv.hpp>

#include <iostream>
#include <thread>

#include "motion_detector.hpp"

using MJPEGStreamer = nadjieb::MJPEGStreamer;



int main(int argc, char** argv)
{
    MJPEGStreamer streamer;
    streamer.start(8080);

    cv::VideoCapture vcap(0);

    int frame_count = 32;

    if(!vcap.open(0)){
        std::cout << "Error opening video stream." << std::endl;
        return -1;
    }

    cv::Mat image, gray_img;
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 90};

    SingleMotionDetector md(0.1);
    int total = 0;

    while (streamer.isRunning()){
		cv::Mat frame;
        vcap >> frame;

		// cv::imshow("img", frame);
        if(frame.empty()){
            std::cout << "not getting image\n";
            continue;
        }

		cv::cvtColor(frame, gray_img, cv::COLOR_BGR2GRAY, 0);
		cv::GaussianBlur(gray_img, gray_img, cv::Size(7, 7), 0);

		if( total > frame_count){
			auto [thresh_img, minX, minY, maxX, maxY] = md.detect(gray_img);

			if(!thresh_img.empty()){
				//std::cout << "detected contours: " << minX << " " << minY << " " << maxX << " " << maxY << "\n";
				cv::rectangle(frame, cv::Rect(cv::Point2i(minX, minY), cv::Point2i(maxX, maxY)), cv::Scalar(0, 0, 255), 2);
			}
		}

		md.update(gray_img);
		total++;

        std::vector<uchar> buff_bgr;

		cv::imencode(".jpg", frame, buff_bgr, params);


        streamer.publish("/bgr", std::string(buff_bgr.begin(), buff_bgr.end()));

        std::this_thread::sleep_for(std::chrono::milliseconds(10));

    }

    streamer.stop();

    return 0;
}
