#include "CascadeClassifier.h"
#include "EnsembleClassifier.h"
#include "Utils.h"


tld::CascadeClassifier::CascadeClassifier(const cv::Mat &initialFrame,
                                          const BBox &initialBbox,
                                          const ObjectModel &objectModel,
                                          Params* params,
                                          tld::utils::Random* rng)
{
    this->params = params;
    this->objectModel = objectModel;
    this->initialBbox = initialBbox;

    cv::Mat iImage;
    cv::Mat iImageSq;
    tld::utils::computeIntegralImage2(initialFrame, iImage, iImageSq);
    this->varMin = params->VARIANCE_FRACTION * this->patchVariance(iImage, iImageSq, initialBbox);

    // Generate a pool of ensemble classifiers
    const float stepX = params->WIDTH_FRACTION * initialBbox.width;
    const float stepY = params->HEIGHT_FRACTION * initialBbox.height;
    for (float s = params->MIN_SCALE; s <= params->MAX_SCALE; s += params->SCALE_STEP)
    {
        float w = s * initialBbox.width;
        float h = s * initialBbox.height;
        if (w * h >= params->MIN_AREA)
        {
            for (float y = 0.0f; (y + h) <= initialFrame.rows; y += stepY)
            {
                for (float x = 0.0f; (x + w) <= initialFrame.cols; x += stepX)
                {
                    BBox bbox(x, y, w, h);
                    tld::EnsembleClassifier ensClf(params->NUM_FERNS, params->NUM_BINARY_FEATURES, bbox, rng);
                    this->ensClfPool.push_back(ensClf);
                }
            }
        }
    }

    std::cout << "Cascade detector initialized." << std::endl;
}


/**
 * Computes the variance of an image patch defined by the given bbox
 * using integral images integralImage and integralImage2.
 */
float tld::CascadeClassifier::patchVariance(const cv::Mat &integralImage,
                                            const cv::Mat &integralImage2,
                                            const BBox &bbox) const
{
    float N = bbox.width * bbox.height;
    float m = tld::utils::sumPatch(integralImage, bbox) / N;
    float m2 = tld::utils::sumPatch(integralImage2, bbox) / N;

    return m2 - m * m;
}


float tld::CascadeClassifier::templateMatching(const cv::Mat& patch) const
{
    cv::Mat patchResized;
    cv::resize(patch, patchResized, params->TEMPLATE_SIZE, 0, 0, cv::INTER_CUBIC);

    std::vector<float> posDistances;
    for (const cv::Mat& posTempl : this->objectModel.positiveTemplates)
    {
        if (!posTempl.empty())
        {
            float posDist = 1.0f - 0.5f * (tld::utils::computeNCC(patchResized, posTempl) + 1.0f);
            posDistances.push_back(posDist);
        }
    }
    float minPosDist = *std::min_element(posDistances.begin(), posDistances.end());

    std::vector<float> negDistances;
    for (const cv::Mat& negTempl : this->objectModel.negativeTemplates)
    {
        if (!negTempl.empty())
        {
            float negDist = 1.0f - 0.5f * (tld::utils::computeNCC(patchResized, negTempl) + 1.0f);
            negDistances.push_back(negDist);
        }
    }
    float minNegDist = *std::min_element(negDistances.begin(), negDistances.end());

    float relativeDist = minNegDist / (minNegDist + minPosDist);

    return relativeDist;
}


std::vector<BBox> tld::CascadeClassifier::detect(const cv::Mat &frame) const
{
    // Preprocessing.
    // cv::Mat frameBlured;
    // cv::GaussianBlur(frame, frameBlured, cv::Size(0, 0), 3.0, 3.0, 0);
    cv::Mat iImage;
    cv::Mat iImageSq;
    tld::utils::computeIntegralImage2(frame, iImage, iImageSq);

    // Detection loop over all the subwindows.
    std::vector<BBox> detectedBBoxes;
    for (std::size_t i = 0; i < this->ensClfPool.size(); ++i)
    {
        BBox bbox = this->ensClfPool[i].bbox;

        // 1. Variance filtering
        if (this->patchVariance(iImage, iImageSq, bbox) > this->varMin)
        {
            // 2. Ensemble classification
            // if (this->ensClfPool[i].classifyPatch(frameBlured) > 0.5f)
            if (this->ensClfPool[i].classifyPatch(frame) > 0.5f)
            {
                // 3. Template matching
                cv::Mat patch = frame(bbox);
                if (this->templateMatching(patch) > params->THETA_MINUS)
                {
                    detectedBBoxes.push_back(bbox);
                }
                
            }
        }
    }

    // Apply non-maximal suppression on the set of detected bboxes
    std::vector<BBox> detectedBboxesFinal = tld::utils::NMS(detectedBBoxes, params->OVERLAP_THRESHOLD);

    return detectedBboxesFinal;
}