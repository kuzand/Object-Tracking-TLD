#include <opencv2/video/tracking.hpp>
#include <math.h>  // fabs
#include <cmath>  // hypot
#include "MedianFlowTracker.h"
#include "Utils.h"






/**
 * MedianFlowTracker constructor
 */
//tld::MedianFlowTracker::MedianFlowTracker() {}

tld::MedianFlowTracker::MedianFlowTracker(cv::Mat initialFrame, BBox initialBbox, Params* params, tld::utils::Random* rng)
{
    CV_Assert(initialFrame.type() == CV_8UC1);

    this->params = params;
    this->rng = rng;
    this->previousFrame = initialFrame;
    this->previousBbox = initialBbox;
    this->previousPoints = generatePoints(initialBbox);

    std::cout << "Median Flow Tracker initialized." << std::endl;
}


/**
 * Initialize the points for tracking within the initial bbox
 */
std::vector<cv::Point2f> tld::MedianFlowTracker::generatePoints(const BBox& bbox)
{

    std::vector<cv::Point2f> points;

    if (bbox.empty())
    {
        return points;
    }

    // Distribute the points uniformly or randomly over the bbox
    if (params->RAND_POINTS)
    {
        for (int i = 0; i < params->TOTAL_NUM_POINTS; ++i)
        {
            float x = rng->randf(bbox.x, bbox.x + bbox.width);
            float y = rng->randf(bbox.y, bbox.y + bbox.height);
            points.push_back(cv::Point2f(x, y));
        }
    }
    else
    {
        for (int i = 0; i < params->LEN_POINTS; ++i)
        {
            for (int j = 0; j < params->LEN_POINTS; ++j)
            {
                float x = bbox.x + i * (bbox.width / params->LEN_POINTS);
                float y = bbox.y + j * (bbox.height / params->LEN_POINTS);
                points.push_back(cv::Point2f(x, y));
            }
        }
    }

    return points;
}


/**
 * MedianFlowTracker re-initialization, performed during the TLD fusion if the tracker
 * fails or the tracked bbox is less confident than the bbox obtained by the detector.
 */
void tld::MedianFlowTracker::reinitialize(const cv::Mat& frame, const BBox& bbox)
{
    this->previousFrame = frame;
    this->previousBbox = bbox;
    this->previousPoints = generatePoints(bbox);
}


/**
* Updates the forwardStatus vector by computing the Forward-Backward error values.
* The forwardStatus[i] is set to 1 if the FB error of the i-th point is less than
* the median of the FB erors of all the points, otherwise it is set to 0.
*/
void tld::MedianFlowTracker::checkFB(const cv::Mat &newFrame,
                                    const std::vector<cv::Point2f> &newPoints,
                                    std::vector<unsigned char> &forwardStatus)
{
    std::vector<unsigned char> backwardStatus;
	std::vector<float> err;
    std::vector<cv::Point2f> pointsReprojected;
    cv::calcOpticalFlowPyrLK(newFrame, this->previousFrame,
                             newPoints, pointsReprojected,
                             backwardStatus, err,
                             params->LK_WIN_SIZE,
                             params->MAX_PYR_LEVEL,
                             params->TERM_CRITERIA);

    CV_Assert(newPoints.size() == pointsReprojected.size());
    CV_Assert(this->previousPoints.size() == pointsReprojected.size());

    std::vector<float> fbErrors;
    for (std::size_t i = 0; i < pointsReprojected.size(); ++i)
    {
        if (forwardStatus[i] == 1 && backwardStatus[i] == 1)
        {
            cv::Point2f p = this->previousPoints[i] - pointsReprojected[i];
            fbErrors.push_back(std::hypot(p.x, p.y));
        }
        else
        {
            forwardStatus[i] = 0;
        }
    }

    float fbMedian = tld::utils::median(fbErrors);

    if (fbMedian > params->FB_THRESHOLD)
    {
        std::fill(forwardStatus.begin(), forwardStatus.end(), 0);
    }
    else
    {
        int j = 0;
        for (std::size_t i = 0; i < forwardStatus.size(); ++i)
        {
            if (forwardStatus[i] == 1)
            {
                //forwardStatus[i] = (fbErrors[j] <= fbMedian);
                if(fbErrors[j] <= fbMedian)
                {
                    forwardStatus[i] = 1;
                }
                else
                {
                    forwardStatus[i] = 0;
                }
                j++;
            }
        }
    }
}



/**
* Updates the forwardStatus vector based on the NCC values of patches around ther tracked point.
*/
void tld::MedianFlowTracker::checkNCC(const cv::Mat &newFrame,
                                      const std::vector<cv::Point2f> &newPoints,
                                      std::vector<unsigned char> &forwardStatus)
{
    std::vector<float> nccValues;
    for (int i = 0; i < params->TOTAL_NUM_POINTS; ++i)
    {
        if (forwardStatus[i] == 1)
        {
            cv::Mat patch1 = tld::utils::getPatch(this->previousFrame, this->previousPoints[i],
                                                  params->NCC_PATCH_SIZE);
            cv::Mat patch2 = tld::utils::getPatch(newFrame, newPoints[i],
                                                  params->NCC_PATCH_SIZE);
            float ncc = tld::utils::computeNCC(patch1, patch2);
            nccValues.push_back(ncc);
        }
    }

    float medianNCC = tld::utils::median(nccValues);

    int j = 0;
    for (int i = 0; i < params->TOTAL_NUM_POINTS; ++i)
    {  
        if (forwardStatus[i] == 1)
        {
            if (nccValues[j] >= medianNCC)
            {
                forwardStatus[i] = 1;
            }
            else
            {
                forwardStatus[i] = 0;
            }
            j++;
        }
    }
}


/** 
 * Tracking
 */
BBox tld::MedianFlowTracker::track(const cv::Mat &newFrame)
{

    if (this->previousBbox.empty() || this->previousPoints.empty())
    {
        reinitialize(newFrame, BBox());
        return BBox();
    }
    
    // Calculate optical flow using the iterative Lucas-Kanade method with pyramids.
    std::vector<cv::Point2f> newPoints;
    std::vector<unsigned char> forwardStatus;
	std::vector<float> err;
    cv::calcOpticalFlowPyrLK(this->previousFrame, newFrame,
                             this->previousPoints, newPoints,
                             forwardStatus, err,
                             params->LK_WIN_SIZE,
                             params->MAX_PYR_LEVEL,
                             params->TERM_CRITERIA);

    // Compute the forward-backward (FB) errors and update the forwardStatus.
    // At every iteration the FB check cuts the number of points to half.
    tld::MedianFlowTracker::checkFB(newFrame, newPoints, forwardStatus);

    // Compute the normalized correlation coefficient (NCC) and update the forwardStatus.
    tld::MedianFlowTracker::checkNCC(newFrame, newPoints, forwardStatus);

    // Select points that where successfully tracked.
    std::vector<cv::Point2f> trackedPoints;
    std::vector<float> translationsX;
    std::vector<float> translationsY;
    for(std::size_t i = 0; i < newPoints.size(); ++i)
    {
        if(forwardStatus[i] == 1)
        {
            trackedPoints.push_back(newPoints[i]);
            float dx = newPoints[i].x - this->previousPoints[i].x;
            float dy = newPoints[i].y - this->previousPoints[i].y;
            translationsX.push_back(dx);
            translationsY.push_back(dy);
        }
    }

    // Can be empy only if calcOpticalFlowPyrLK fails to track the points, otherwise there will be at least one point.
    if (trackedPoints.size() < 1)
    {
        // Tracking failed, reinitialize the tracker with empty bbox and return empty bbox.
        std::cout << "Tracking failed because LK failed" << std::endl;
        reinitialize(newFrame, BBox());
        return BBox();
    }
    
    // Compute the median translation in x and y directions (needed for the computation of the new bbox).
    float mDx = tld::utils::median(translationsX);
    float mDy = tld::utils::median(translationsY);
  
    // Check for tracking failure by comparing translations to the median translation.
    float dm = std::hypot(mDx, mDy);
    std::vector<float> displacementResiduals;
    for (std::size_t i = 0; i < translationsX.size(); ++i)
    {
        cv::Point2f p = cv::Point2f(translationsX[i], translationsY[i]);
        float di = std::hypot(p.x, p.y);
        displacementResiduals.push_back(std::fabs(di - dm));
    }
    if (tld::utils::median(displacementResiduals) > params->MAX_MEDIAN_DISPLACEMENT)
    {
        // Tracking failed, reinitialize the tracker with empty bbox and return empty bbox.
        std::cout << "Tracking failed because median displacement is too big" << std::endl;
        reinitialize(newFrame, BBox());
        return BBox();
    }

    // Compute the scale factor for the new bbox:
    // First, compute the pairwise distances between all the previousPoints.
    // Similarly, compute the pairwise distances between all the newPoints.
    // Second, compute the ratios between the corresponding distances.
    // Finaly, use the median of the ratios as the change in the scale of the new bbox.
    std::vector<float> scales;
    for (std::size_t i = 0; i < newPoints.size(); ++i)
    {
        if(forwardStatus[i] == 1)
        {
            for (std::size_t j = i + 1; j < newPoints.size(); ++j)
            {
                if(forwardStatus[j] == 1)
                {
                    cv::Point2f p1 = this->previousPoints[i] - this->previousPoints[j];
                    cv::Point2f p2 = newPoints[i] - newPoints[j];
                    float s1 = std::hypot(p1.x, p1.y);
                    float s2 = std::hypot(p2.x, p2.y);
                    if (s1 != 0 && s2 != 0)
                    {
                        scales.push_back(s2 / s1);
                    }
                }
            }
        }
    }

    float newBboxScale = 1.0f;
    if (!scales.empty())
    {
        newBboxScale = tld::utils::median(scales);
    }

    BBox newBbox;
    newBbox.x = this->previousBbox.x + mDx;
    newBbox.y = this->previousBbox.y + mDy;
    newBbox.width = newBboxScale * this->previousBbox.width;
    newBbox.height = newBboxScale * this->previousBbox.height;
    //newBbox &= BBox(0, 0, newFrame.cols, newFrame.rows);
    if (!tld::utils::bboxWithinImage(newBbox, newFrame))
    {
        std::cout << "bbox crossed the boundaries" << std::endl;
        reinitialize(newFrame, BBox());
        return BBox();
    }

    //if (newBBox.empty())
    if (newBbox.width <= 5 || newBbox.height <= 5)
    {
        std::cout << "bbox too small" << std::endl;
        reinitialize(newFrame, BBox());
        return BBox();
    }

    // Reinitialize the tracker
    reinitialize(newFrame, newBbox);

    return newBbox;
}
