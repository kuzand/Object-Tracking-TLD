#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace tld
{
    struct Params
    {
        // Constructor
        Params();

        // Read params from a file
        void write(const std::string& filename) const;

        // Write params to a file
        void read(const std::string& filename);

        // Print all the parameters
        void printParams() const;

        int RNG_SEED;

        // Median Flow tracker parameters
        int LEN_POINTS;                  // number of points in a single dimension inside the bbox
        int TOTAL_NUM_POINTS;            // LEN_POINTS * LEN_POINTS
        bool RAND_POINTS;                // flag to randomly distribute the points in bbox
        cv::Size LK_WIN_SIZE;            // window size parameter for Lucas-Kanade optical flow
        int MAX_PYR_LEVEL;               // maximal pyramid level number for Lucas-Kanade optical flow
        cv::Size NCC_PATCH_SIZE;         // patch size around a point for computing normalized cross-correlation
        float FB_THRESHOLD;              // threshold for the median forward-backward error
        float MAX_MEDIAN_DISPLACEMENT;   // used for detection of tracking failure
        cv::TermCriteria TERM_CRITERIA;  // termination criteria for Lucas-Kanade optical flow

        // Cascade classifier parameters
        float VARIANCE_FRACTION;  // variance fraction of the initial patch (used for variance filter)
        float OVERLAP_THRESHOLD;  // for non-maximal suppression
        int NUM_FERNS;            // number of ferns of ensemble classifier
        int NUM_BINARY_FEATURES;  // number of binary features (pixel comparisons) of each fern
        float SCALE_STEP;         // sliding window scale step
        float MIN_SCALE;          // min sliding window scale
        float MAX_SCALE;          // max sliding window scale
        float WIDTH_FRACTION;     // used for the computation of the sliding window step X
        float HEIGHT_FRACTION;    // used for the computation of the sliding window step Y
        float MIN_AREA;           // minimum area of the sliding window
        float THETA_PLUS;         // threshold for positive patch classification
        float THETA_MINUS;        // threshold for negative patch classification

        // Object model parameters
        bool RAND_REPLACEMENT;
        cv::Size TEMPLATE_SIZE; // object model template size
        size_t INIT_OBJ_MODEL_SIZE;
        size_t MAX_OBJ_MODEL_SIZE;
    };
} // namespace tld