#include <opencv2/highgui.hpp>
#include <tuple>

class SingleMotionDetector
{
public:
    SingleMotionDetector(double accumWeight=0.5);
    ~SingleMotionDetector();

public:
    void update(cv::Mat& image);
    std::tuple<cv::Mat, int, int, int, int> detect(cv::Mat& image, int tVal=25);


private:
    double m_accuWeight;
    cv::Mat m_bg;
};
