#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <utility>  // std::pair
#include "Utils.h"


using BBox = cv::Rect2f;

namespace tld
{
class Fern
{
public:
    int numBinaryFeatures;
    std::vector<cv::Point2i> pixelPairs;
    std::vector<float> numPos;
    std::vector<float> numNeg;
    
public:
    Fern(int numBinaryFeatures, const BBox& bbox, tld::utils::Random* rng);
    int calcFern(const cv::Mat& frame, const BBox& bbox) const;
};


class EnsembleClassifier
{
public:
    int numFerns;       // number of ferns in the ensemble
    int numBinaryFeatures;  // number of binary features (pixel pairs) per fern
    BBox bbox;
    std::vector<Fern> ferns;

    EnsembleClassifier(int numFerns, int numBinaryFeatures, BBox bbox, tld::utils::Random* rng);
    float classifyPatch(const cv::Mat& frame) const;    
};

} // namespace tld