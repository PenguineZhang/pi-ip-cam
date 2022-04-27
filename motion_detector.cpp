#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <vector>

#include "motion_detector.hpp"

SingleMotionDetector::SingleMotionDetector(double accumWeight)
{
    m_accuWeight = accumWeight;
}

SingleMotionDetector::~SingleMotionDetector(){}

void SingleMotionDetector::update(cv::Mat& image)
{
    if(m_bg.empty()){
        m_bg = image;
		m_bg.convertTo(m_bg, CV_64FC1);
        return;
    }

    cv::accumulateWeighted(image, m_bg, m_accuWeight);
}

std::tuple<cv::Mat, int, int, int, int> SingleMotionDetector::detect(cv::Mat& image, int tVal)
{
    cv::Mat delta, thresh_img, img_8bit;

    m_bg.convertTo(img_8bit, CV_8UC1);
    cv::absdiff(img_8bit, image, delta);

    double thresh_val = cv::threshold(delta, thresh_img, tVal, 255, cv::THRESH_BINARY);

    int erosion_size = 0;
    int dilation_size = 0;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(0, 0));
    cv::erode(thresh_img, thresh_img, element, cv::Point(-1, -1), 2);
    cv::dilate(thresh_img, thresh_img, element, cv::Point(-1, -1), 2);

    std::vector<std::vector<cv::Point>> cnts;
    cv::findContours(thresh_img, cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if(cnts.size() == 0){
        //std::cout << "no contour is found, return nothing\n";
        return std::tuple<cv::Mat, int, int, int, int>();
    }

	int minX=9999, minY=9999, maxX=-1, maxY=-1;
	for(int i = 0; i < cnts.size(); ++i){
		cv::Rect rect = cv::boundingRect(cnts[i]);
		minX = std::min(rect.x, minX);
		minY = std::min(rect.y, minY);
		maxX = std::max(maxX, rect.x + rect.width);
		maxY = std::max(maxY, rect.y + rect.height);
	}

    return std::make_tuple(thresh_img,minX, minY, maxX, maxY);

}
