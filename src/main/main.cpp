#include <cstdlib>
#include <ctime>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

cv::Mat generateDaphnia(cv::Mat& dst, const cv::Scalar& color) {
    // initial size: 5x10
    int width = std::rand() % 3 + 4;        // from 4 to 6
    int height = std::rand() % 4 + 8;       // from 8 to 11
    int x_coord = std::rand() % 45 + 10;    // from 10 to 54
    int y_coord = std::rand() % 45 + 10;    // from 10 to 54
    size_t angle = std::rand() % 181;       // from 0 to 180

    cv::Mat daphnia_mask = cv::Mat::zeros(dst.size(), CV_32FC1);

    daphnia_mask += 255;
    cv::ellipse(daphnia_mask, { x_coord, y_coord }, { width, height }, angle, 0, 360, color, -1);
    cv::GaussianBlur(daphnia_mask, dst, {5, 5}, 10);
    cv::threshold(daphnia_mask, daphnia_mask, 150, 255, cv::THRESH_BINARY_INV);
    daphnia_mask /= 255;

    std::cout << width << " " << height << " " << x_coord << " " << y_coord << " " << angle << std::endl;
    return daphnia_mask;
}

void generateFrame(cv::Mat& dst,
    size_t median_background, size_t sigma_background,
    const cv::Scalar& daphnia_color, const std::string& filename = "null") {

    //background generation
    cv::randn(dst, median_background, sigma_background);

    //daphnia generation
    size_t quantity = std::rand() % 3 + 1;
    cv::Mat daphnia_mask = cv::Mat::zeros(dst.size(), CV_32FC1);
    cv::Mat daphnia_inv_mask = cv::Mat::zeros(dst.size(), CV_32FC1);
    cv::Mat daphina_object = cv::Mat::zeros(dst.size(), CV_32FC1);
    for (size_t i{}; i < quantity; ++i) {
        daphnia_mask = generateDaphnia(daphina_object, daphnia_color);
        cv::threshold(daphnia_mask, daphnia_inv_mask, 0.5, 1, cv::THRESH_BINARY_INV);
        dst.convertTo(dst, CV_32FC1);
        dst = daphnia_inv_mask.mul(dst);            //clear area for daphnia
        dst += daphnia_mask.mul(daphina_object);    //write generated daphnia
        dst.convertTo(dst, CV_8UC1);
    }

    //general noise
    cv::Mat general_noise = cv::Mat::zeros(dst.size(), dst.type());
    cv::randn(general_noise, 0, sigma_background);
    dst += general_noise;

    //writting
    if (filename != "null")
        cv::imwrite(filename, dst);
}

void calculateHeatmap(const cv::Mat& src, cv::Mat& dst,
    const cv::Size& gauss_size, double gauss_sigma,
    size_t median_size1, size_t median_size2,
    const std::string& filename = "null") {

    //denoising
    cv::Mat denoised = cv::Mat(src.rows, src.cols, CV_8UC1);
    cv::GaussianBlur(src, denoised, gauss_size, gauss_sigma);

    //gitting background color
    cv::Mat background = cv::Mat(src.rows, src.cols, CV_8UC1);
    cv::medianBlur(denoised, background, median_size1);
    cv::medianBlur(background, background, median_size2);

    //getting heatmap
    double min, max;
    dst = background - denoised;
    cv::minMaxIdx(dst, &min, &max);
    dst.convertTo(dst, CV_32FC1);
    dst /= max;
    std::cout << min << " " << max << std::endl;

    if (filename != "null")
        cv::imwrite(filename, dst, { cv::IMWRITE_EXR_TYPE_FLOAT });

}

int main() {
    std::srand((unsigned)std::time(0));

    //cv::Mat videoframe = cv::imread("../../../other/frame.png", cv::IMREAD_GRAYSCALE);
    cv::Mat frame = cv::Mat::zeros(64, 64, CV_8UC1);
    cv::Mat heatmap = cv::Mat::zeros(frame.rows, frame.cols, CV_32FC1);

    //for (size_t i{ 1 }; i < 6; ++i) {
    //    //frame generation
    //    generateFrame(frame, 180, 3, 20, "../../../data/src" + std::to_string(i) + ".png");
    //    //heatmap calculation
    //    calculateHeatmap(frame, heatmap, { 11, 11 }, 80, 41, 41, "../../../data/dst" + std::to_string(i) + ".png");
    //}

    generateFrame(frame, 180, 3, 100);
    calculateHeatmap(frame, heatmap, { 11, 11 }, 80, 41, 41);

    cv::waitKey(0);
}
