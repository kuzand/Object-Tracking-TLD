#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "Params.h"
#include "Utils.h"


using BBox = cv::Rect2f;

namespace tld
{

    class MedianFlowTracker
    {

    public:
        Params* params;

        MedianFlowTracker() = default;
        MedianFlowTracker(cv::Mat initialFrame, BBox initialBbox, Params* params, tld::utils::Random* rng);

        BBox track(const cv::Mat &newFrame);
        void reinitialize(const cv::Mat& frame, const BBox& bbox);
   
    private:
        tld::utils::Random* rng;

        cv::Mat previousFrame;
        std::vector<cv::Point2f> previousPoints;
        BBox previousBbox;

        std::vector<cv::Point2f> generatePoints(const BBox& bbox);

        // Forward-Backward (FB) error
        void checkFB(const cv::Mat &newFrame,
                     const std::vector<cv::Point2f> &newPoints,
                     std::vector<unsigned char> &forwardStatus);

        // Normalized correlation coefficient (NCC)
        void checkNCC(const cv::Mat &newFrame,
                      const std::vector<cv::Point2f> &newPoints,
                      std::vector<unsigned char> &forwardStatus);

    };

} // namespace tld