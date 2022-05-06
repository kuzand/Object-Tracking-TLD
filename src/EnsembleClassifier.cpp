#include "EnsembleClassifier.h"
#include "Utils.h"


/**
* Constructor of Fern.
*/
tld::Fern::Fern(int numBinaryFeatures, const BBox& bbox, tld::utils::Random* rng)
{
    //CV_Assert(numBinaryFeatures > 0  && !bbox.empty());

    this->numBinaryFeatures = numBinaryFeatures;
    this->pixelPairs = std::vector<cv::Point2i>(2 * numBinaryFeatures);

    int posteriorSize = (std::uint64_t(1) << numBinaryFeatures);  // 2^numBinaryFeatures
    this->numPos = std::vector<float>(posteriorSize, 0.0f);
    this->numNeg = std::vector<float>(posteriorSize, 0.0f);

    // Generate random pixel-pairs locations
    for (int i = 0; i < 2 * numBinaryFeatures; i += 2)
    {
        int x1 = std::floor(rng->randi(bbox.x, bbox.x + bbox.width));
        int y1 = std::floor(rng->randi(bbox.y, bbox.y + bbox.height));
        int x2 = std::floor(rng->randi(bbox.x, bbox.x + bbox.width));
        int y2 = std::floor(rng->randi(bbox.y, bbox.y + bbox.height));
        cv::Point2i p1(x1, y1);
        cv::Point2i p2(x2, y2);

        this->pixelPairs[i] = p1;
        this->pixelPairs[i + 1] = p2;
    }
}


/**
* Calculates the fern value for a given frame.
*/
int tld::Fern::calcFern(const cv::Mat& frame, const BBox& bbox) const
{
    CV_Assert(tld::utils::bboxWithinImage(bbox, frame));

    int F = 0;
    for (int i = 0; i < 2 * numBinaryFeatures; i += 2)
    {
        cv::Point2i p1 = this->pixelPairs[i];
        cv::Point2i p2 = this->pixelPairs[i + 1];
        uchar pix1 = frame.at<uchar>(p1.y, p1.x);
        uchar pix2 = frame.at<uchar>(p2.y, p2.x);
        F = (F << 1) | (pix1 > pix2);
    }

    return F;
}


/**
* Constructor of EnsembleClassifier.
*/
tld::EnsembleClassifier::EnsembleClassifier(int numFerns, int numBinaryFeatures, BBox bbox, tld::utils::Random* rng)
{

    CV_Assert(numFerns > 0 && numBinaryFeatures > 0 && !bbox.empty());

    this->numFerns = numFerns;
    this->numBinaryFeatures = numBinaryFeatures;
    this->bbox = bbox;

    for (int k = 0; k < this->numFerns; ++k)
    {
        this->ferns.emplace_back(Fern(numBinaryFeatures, bbox, rng));
    }
}


/**
* Returns the posterior probability of the given patch (frame(this->bbox)) being positive.
*/
float tld::EnsembleClassifier::classifyPatch(const cv::Mat& frame) const
{
    // Average posterior probability from all the ferns.
    float avgP = 0.0f;
    for (int k = 0; k < this->numFerns; ++k)
    {
        int Fk = this->ferns[k].calcFern(frame, this->bbox);
        float numPk = this->ferns[k].numPos[Fk];
        float numNk = this->ferns[k].numNeg[Fk];
        if ((numPk + numNk) != 0)
        {
            avgP += numPk / (numPk + numNk);
        }
    }
    avgP /= this->numFerns;

    return avgP;
}