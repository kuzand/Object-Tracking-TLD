#define _USE_MATH_DEFINES
#include <cmath>      // M_PI, std::hypot, std::cos, std::sin
#include <functional>  // std::bind
#include "ObjectModel.h"
#include "Utils.h"


const float PI = static_cast<float>(M_PI);

tld::ObjectModel::ObjectModel(const cv::Mat& initialFrame,
                              const BBox& initialBbox,
                              Params* params,
                              tld::utils::Random* rng)
{
    CV_Assert(tld::utils::bboxWithinImage(initialBbox, initialFrame));

    this->params = params;
    this->rng = rng;
    cv::Mat positivePatch = initialFrame(initialBbox);

    // Create warps of the initial positive template.
    for (size_t i = 0; i < params->INIT_OBJ_MODEL_SIZE; ++i)
    {
        //float angle = rng->randf(-3.0f, 3.0f);
        //float scale = rng->randf(0.5f, 2.0f);
        //cv::Mat positivePatchWarped = this->warpImage(positivePatch, angle, scale);
        //cv::Mat positivePatchWarpedNoisy = this->addGaussianNoise(positivePatchWarped, 0.0f, 3.0f);
        //cv::Mat positivePatchWarpedNoisyResized;
        //cv::resize(positivePatchWarpedNoisy, positivePatchWarpedNoisyResized, params->TEMPLATE_SIZE, 0, 0, cv::INTER_CUBIC);
        //this->positiveTemplates.push_back(positivePatchWarpedNoisyResized);

        float xShift = initialBbox.width * 0.2f;
        float yShift = initialBbox.height * 0.2f;
        BBox posShiftedBbox(initialBbox.x - rng->randf(-xShift, xShift),
                            initialBbox.y - rng->randf(-yShift, yShift),
                            initialBbox.width,
                            initialBbox.height);
        cv::Mat posShiftedBboxPatch = initialFrame(posShiftedBbox & BBox(0, 0, initialFrame.cols, initialFrame.rows));
        cv::Mat posShiftedBboxPatchResized;
        cv::resize(posShiftedBboxPatch, posShiftedBboxPatchResized, params->TEMPLATE_SIZE, 0, 0, cv::INTER_CUBIC);
        this->positiveTemplates.push_back(posShiftedBboxPatchResized);
    }

    // Create random negative patches outside of the initial positive patch.
    float r = std::hypot(initialBbox.width, initialBbox.height);
    float dr = 0.2f * r;
    for (size_t i = 0; i < params->INIT_OBJ_MODEL_SIZE; ++i)
    {
        BBox negNearBbox = this->createNearbyBbox(initialBbox, 0.2f, r, dr, 0.3f);
        if (negNearBbox.area() >= params->MIN_AREA)
        {
            cv::Mat negNearBboxPatch = initialFrame(negNearBbox & BBox(0, 0, initialFrame.cols, initialFrame.rows));
            cv::Mat negNearBboxPatchResized;
            cv::resize(negNearBboxPatch, negNearBboxPatchResized, params->TEMPLATE_SIZE, 0, 0, cv::INTER_CUBIC);
            this->negativeTemplates.push_back(negNearBboxPatchResized);
        }
    }

    std::cout << "Object model initialized." << std::endl;
}


void tld::ObjectModel::addPositiveTemplate(cv::Mat positiveTemplate)
{

    if (this->positiveTemplates.size() < params->MAX_OBJ_MODEL_SIZE)
    {
        this->positiveTemplates.push_back(positiveTemplate);
    }
    else
    {
        if (params->RAND_REPLACEMENT)
        {
            int randIndex = rng->randi(0, this->positiveTemplates.size() - 1);
            this->positiveTemplates[randIndex] = positiveTemplate;
        }
        else
        {
            this->positiveTemplates.pop_front();
            this->positiveTemplates.push_back(positiveTemplate);
        }
    }
}


void tld::ObjectModel::addNegativeTemplate(cv::Mat negativeTemplate)
{
    if (this->negativeTemplates.size() < params->MAX_OBJ_MODEL_SIZE)
    {
        this->negativeTemplates.push_back(negativeTemplate);

    }
    else
    {
        if (params->RAND_REPLACEMENT)
        {
            int randIndex = rng->randi(0, this->negativeTemplates.size() - 1);
            this->negativeTemplates[randIndex] = negativeTemplate;
        }
        else
        {
            this->negativeTemplates.pop_front();
            this->negativeTemplates.push_back(negativeTemplate);
        }

    }
}


/**
 * Randomly generates a point inside the given bbox.
 */
cv::Point2f tld::ObjectModel::getRandomPointInsideBbox(const BBox& bbox)
{
    float x = rng->randf(bbox.x, bbox.x + bbox.width);
    float y = rng->randf(bbox.y, bbox.y + bbox.height);
    return cv::Point2f(x, y);
}


/**
 * Adds Gussian noise to the input image.
 */
cv::Mat tld::ObjectModel::addGaussianNoise(const cv::Mat& img, float mean, float sigma)
{

    CV_Assert(img.type() == CV_8UC1);

    int nRows = img.rows;
    int nCols = img.cols;

    cv::Mat imgNoisy = cv::Mat::zeros(img.size(), img.type());

    for (int i = 0; i < nRows; ++i)
    {
        uchar* const imgNoisyRowPtr = imgNoisy.ptr<uchar>(i);
        const uchar* const imgRowPtr = img.ptr<uchar>(i);
        for (int j = 0; j < nCols; ++j)
        {
            float r = std::roundf(rng->randN(mean, sigma));
            int val = static_cast<int>(imgRowPtr[j] + r);
            imgNoisyRowPtr[j] = static_cast<int>(std::clamp(val, 0, 255));
        }
    }

    return imgNoisy;
}


/**
 * Randomly creates a bbox nearby the given bbox.
 * r and theta specifiy the aspect ratio.
 */
BBox tld::ObjectModel::createNearbyBbox(const BBox& bbox,
                                        float marginFrac,
                                        float r,
                                        float dr,
                                        float dtheta)
{
    float x = bbox.x;
    float y = bbox.y;
    float w = bbox.width;
    float h = bbox.height;
    float marginX = w * marginFrac;
    float marginY = h * marginFrac;
    float xc = x + marginX;
    float yc = y + marginY;
    float wc = w - 2 * marginX;
    float hc = h - 2 * marginY;
    float theta = std::atan2(h, w);

    std::vector<cv::Rect2f> bboxes{
        cv::Rect2f(x, y, marginX, marginY),
        cv::Rect2f(xc, y, wc, marginY),
        cv::Rect2f(xc + wc, y, marginX, marginY),
        cv::Rect2f(x, yc, marginX, hc),
        cv::Rect2f(xc + wc, yc, marginX, hc),
        cv::Rect2f(x, yc + hc, marginX, marginY),
        cv::Rect2f(xc, yc + hc, wc, marginY),
        cv::Rect2f(xc + wc, yc + hc, marginX, marginY)
    };

    std::vector<std::vector<float>> directions{
        {theta, PI - theta, PI + theta},
        {theta, PI - theta},
        {theta, PI - theta, -theta},
        {PI - theta, PI + theta},
        {theta, -theta},
        {PI - theta, PI + theta, -theta},
        {PI + theta, -theta, },
        {theta, -theta, PI + theta}
    };

    // Randomly select a bbox from bboxes
    int bboxId = rng->randi(0, 7);
    // Randomly select a point from the selected bbox
    cv::Point2f p1 = getRandomPointInsideBbox(bboxes[bboxId]);
    // Randomly select a direction
    int numDirs = directions[bboxId].size();
    //int dirInd = tld::utils::randUi(0, numDirs - 1);
    int dirInd = rng->randi(0, numDirs - 1);
    float dir = directions[bboxId][dirInd];

    // Find coordinates of the second point around (r, theta)
    float angle = rng->randf(dir - dtheta, dir + dtheta);
    float radius = rng->randf(r - dr, r + dr);
    cv::Point2f p2(p1.x + radius * std::cos(angle), p1.y - radius * std::sin(angle));

    // Create a new bbox based on the two selected points
    float minX = std::min(p1.x, p2.x);
    float maxX = std::max(p1.x, p2.x);
    float minY = std::min(p1.y, p2.y);
    float maxY = std::max(p1.y, p2.y);
    cv::Rect2f newBbox(minX, minY, maxX - minX, maxY - minY);

    return newBbox;
}


/**
* Rotate and scale the input image.
* The center of rotation is the center of the image and the angle is in given in degrees.
*/
cv::Mat tld::ObjectModel::warpImage(const cv::Mat& image, float angle, float scale)
{

    float cos = scale * std::cos(angle * PI / 180.0f);
    float sin = scale * std::sin(angle * PI / 180.0f);
    cv::Mat_<float> rotMat = (cv::Mat_<float>(2, 3) << cos, sin, 0.0f,
                                                      -sin, cos, 0.0f);

    // Image corners
    cv::Point2f pA = cv::Point2f(0.0f, 0.0f);
    cv::Point2f pB = cv::Point2f(image.cols, 0.0f);
    cv::Point2f pC = cv::Point2f(image.cols, image.rows);
    cv::Point2f pD = cv::Point2f(0.0f, image.rows);

    std::vector<cv::Point2f> pts = { pA, pB, pC, pD };
    std::vector<cv::Point2f> ptsTransf;
    cv::transform(pts, ptsTransf, rotMat);


    auto compareCoords = [](cv::Point2f p1, cv::Point2f p2, char coord)
    {
        if (coord == 'x')
            return p1.x < p2.x;

        return p1.y < p2.y;
    };

    using namespace std::placeholders;  // for _1, _2
    float minX = std::min_element(ptsTransf.begin(), ptsTransf.end(),
                                  std::bind(compareCoords, _1, _2, 'x'))->x;
    float maxX = std::max_element(ptsTransf.begin(), ptsTransf.end(),
                                  std::bind(compareCoords, _1, _2, 'x'))->x;
    float minY = std::min_element(ptsTransf.begin(), ptsTransf.end(),
                                  std::bind(compareCoords, _1, _2, 'y'))->y;
    float maxY = std::max_element(ptsTransf.begin(), ptsTransf.end(),
                                  std::bind(compareCoords, _1, _2, 'y'))->y;

    float newW = maxX - minX;
    float newH = maxY - minY;

    cv::Mat_<float> transMat = (cv::Mat_<float>(2, 3) << 0, 0, -minX,
        0, 0, -minY);
    cv::Mat_<float> M = rotMat + transMat;

    cv::Mat warpedImage;
    cv::warpAffine(image, warpedImage, M, cv::Size(newW, newH));

    return warpedImage;
}


