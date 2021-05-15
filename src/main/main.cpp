#include <cstdlib>
#include <ctime>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

void generateDaphnia(cv::Mat& dst, const cv::Scalar& color) {
    // initial size: 5x10
    int width = std::rand() % 3 + 4;        // from 4 to 6
    int height = std::rand() % 4 + 8;       // from 8 to 11
    int x_coord = std::rand() % 45 + 10;    // from 10 to 54
    int y_coord = std::rand() % 45 + 10;    // from 10 to 54
    size_t angle = std::rand() % 181;       // from 0 to 180

    cv::ellipse(dst, { x_coord, y_coord }, { width, height }, angle, 0, 360, color, -1);
    std::cout << width << " " << height << " " << x_coord << " " << y_coord << " " << angle << std::endl;
}

void generateFrame(cv::Mat& dst, const cv::Scalar& color) {
    size_t quantity = std::rand() % 3 + 1;
    for (size_t i{}; i < quantity; ++i)
        generateDaphnia(dst, color);
}

int main() {
    std::srand((unsigned)std::time(0));
    //    cv::Mat frame = cv::imread("../../../data/frame.png", cv::IMREAD_GRAYSCALE);
    cv::Mat mat = cv::Mat::zeros(64, 64, CV_8UC1);
    generateFrame(mat, 255);

    cv::Mat heatmap = cv::Mat(mat.rows, mat.cols, CV_32FC1);
    cv::GaussianBlur(mat, heatmap, {11, 11}, 20);

    cv::waitKey(0);
}
