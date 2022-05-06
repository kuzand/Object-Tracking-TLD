#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>
#include "Params.h"
#include "Utils.h"


using BBox = cv::Rect2f;

namespace tld
{
class ObjectModel
{
public:
    Params* params;

    std::deque<cv::Mat> positiveTemplates;

    std::deque<cv::Mat> negativeTemplates;

    ObjectModel() = default;

    ObjectModel(const cv::Mat &initialFrame,
                const BBox &initialBbox,
                Params* params,
                tld::utils::Random* rng);

    void addPositiveTemplate(cv::Mat positiveTemplate);

    void addNegativeTemplate(cv::Mat negativeTemplate);

private:
    tld::utils::Random* rng;
    
    cv::Point2f getRandomPointInsideBbox(const BBox& bbox);

    BBox createNearbyBbox(const BBox& bbox, float marginFrac, float r, float rFrac, float dtheta);

    cv::Mat addGaussianNoise(const cv::Mat& img, float mean, float sigma);

    cv::Mat warpImage(const cv::Mat& image, float angle, float scale);
};
}  // namespace tld