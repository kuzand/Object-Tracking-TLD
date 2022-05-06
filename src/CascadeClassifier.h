#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "EnsembleClassifier.h"
#include "ObjectModel.h"
#include "Params.h"


using BBox = cv::Rect2f;

namespace tld
{
class CascadeClassifier
{
private:
    BBox initialBbox;
    float varMin;
    
public:
    Params* params;
    ObjectModel objectModel;
    std::vector<tld::EnsembleClassifier> ensClfPool;

public:
    CascadeClassifier() = default;
    CascadeClassifier(const cv::Mat &initialFrame,
                      const BBox &initialBbox,
                      const ObjectModel& objectModel,
                      Params* params,
                      tld::utils::Random* rng);

    std::vector<BBox> detect(const cv::Mat &frame) const;

    float patchVariance(const cv::Mat& integralImage,
                        const cv::Mat& integralImage2,
                        const BBox& bbox) const;

    float templateMatching(const cv::Mat& patch) const;

};

} // namespace tld