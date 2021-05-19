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

void generateFrame(cv::Mat& dst,
    size_t median_background, size_t sigma_background,
    const cv::Scalar& daphnia_color, const std::string& filename = "null") {

    //background generation
    cv::randn(dst, median_background, sigma_background);

    //daphnia generation
    size_t quantity = std::rand() % 3 + 1;
    for (size_t i{}; i < quantity; ++i)
        generateDaphnia(dst, daphnia_color);

    //general noise
    cv::Mat general_noise = cv::Mat::zeros(dst.size(), dst.type());
    cv::randn(general_noise, 0, sigma_background);
    dst += general_noise;

    //writting
    if (filename != "null")
        cv::imwrite(filename, dst);
}

int main() {
    std::srand((unsigned)std::time(0));

    //frame generation
    //cv::Mat frame = cv::imread("../../../data/frame.png", cv::IMREAD_GRAYSCALE);
    cv::Mat mat = cv::Mat::zeros(64, 64, CV_8UC1);
    generateFrame(mat, 180, 3, 20, "../../../data/src1.png");

    //heatmap calculation
    //denoising
    cv::Mat denoised = cv::Mat(mat.rows, mat.cols, CV_8UC1);
    cv::GaussianBlur(mat, denoised, {11, 11}, 20);

    //gitting background color
    cv::Mat background = cv::Mat(mat.rows, mat.cols, CV_8UC1);
    cv::medianBlur(denoised, background, 41);
    cv::medianBlur(background, background, 41);

    //getting heatmap
    double min, max;
    cv::Mat heatmap = background - denoised;
    cv::minMaxIdx(heatmap, &min, &max);
    heatmap.convertTo(heatmap, CV_32FC1);
    heatmap /= max;
    cv::imwrite("../../../data/dst1.png", heatmap, { cv::IMWRITE_EXR_TYPE_FLOAT });

    std::cout << min << " " << max << std::endl;

    cv::waitKey(0);
}
