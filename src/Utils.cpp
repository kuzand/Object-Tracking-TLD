#include <algorithm>  // std::nth_element, std::min_element, std::max_element, std::clamp
#define _USE_MATH_DEFINES
#include <cmath>      // M_PI, std::hypot, std::arctan2, std::cos, std::sin
#include <random>
#include <sstream>
#include <fstream>
#include <limits>
#include <queue>
#include <set>
#include "Utils.h"


const float PI = static_cast<float>(M_PI);

tld::utils::Random::Random(unsigned int seed)
{
    if (seed != NULL)
    {
        this->seed = seed;
    }
    else
    {
        std::random_device seedGenerator;
        this->seed = seedGenerator();
    }

    this->engine.seed(this->seed);
}

/**
 * Generates random floating-point values uniformly distributed in the interval [low, high)
 */
float tld::utils::Random::randf(float low, float high)
{
    std::uniform_real_distribution<float> distribution(low, high);
    return distribution(engine);  // [low, high)
}

/**
 * Generates random integer values uniformly distributed in the closed interval [low, high]
 */
int tld::utils::Random::randi(int low, int high)
{
    std::uniform_int_distribution<int> distribution(low, high);
    return distribution(engine);  // [low, high]
}


/**
 * Generates random numbers according to the Normal (Gaussian) random number distribution.
 */
float tld::utils::Random::randN(float mean, float sigma)
{
    std::normal_distribution<float> distribution(mean, sigma);
    return distribution(engine);
}


/**
 * Rounds a float number x to the given number of decimals n.
 */
float tld::utils::round(float x, unsigned int n)
{
    return std::round(x * std::pow(10, n)) / std::pow(10, n);
}


/**
 * Computes the median of a given vector of float values.
 */
float tld::utils::median(std::vector<float> values)
{
    if (values.empty())
    {
        return 0.0f;
    }
    
    int midIndex = static_cast<int>(values.size() / 2);
    std::nth_element(values.begin(), values.begin() + midIndex, values.end());

    float midVal = values[midIndex];

    if (values.size() % 2 == 0)
    {
        return (midVal + *std::max_element(values.begin(), values.begin() + midIndex)) * 0.5f;
    }
    else
    {
        return midVal;
    }
}


/**
 * Checks if the patch is within the image boundaries.
 */
bool tld::utils::bboxWithinImage(const BBox& bbox, const cv::Mat& image)
{
    CV_Assert(!image.empty() && !bbox.empty());

    float x = bbox.x, y = bbox.y, width = bbox.width, height = bbox.height;

    if (x >= 0 && (x + width) <= image.cols && y >= 0 && (y + height) <= image.rows)
    {
        return true;
    }
    return false;
}


/**
 * Returns image patch specified by its size and the location of its center.
 */
cv::Mat tld::utils::getPatch(const cv::Mat& image, cv::Point2f patchCenter, cv::Size patchSize)
{
    cv::Mat patch;
    cv::Point2f patchTopLeftPoint(patchCenter.x - patchSize.width / 2.0f,
                                 patchCenter.y - patchSize.height / 2.0f);
    BBox patchRect(patchTopLeftPoint, patchSize);

    if (tld::utils::bboxWithinImage(patchRect, image))
    {
        patch = image(patchRect);
    }
    else
    {
        patchRect &= BBox(0, 0, image.cols, image.rows);  // union
        patch = image(patchRect);
    }

    return patch;
}


/**
 * Computes Normalized Correlation Coefficient (NCC) according to the formula:
 *    NCC = (1 / N) * sum[(patch1 - mean1) * (patch2 - mean2) / (sigma1 * sigma2)]
 * The values are in the range -1 to 1.
 */
float tld::utils::computeNCC(const cv::Mat& patch1, const cv::Mat& patch2)
{
    // float ncc;
    // if (patch1.size() == patch2.size())
    // {
    //     int N = patch1.rows * patch2.cols;
    //     cv::Scalar mean1, sigma1;
    //     cv::Scalar mean2, sigma2;
    //     cv::meanStdDev(patch1, mean1, sigma1);
    //     cv::meanStdDev(patch2, mean2, sigma2);

    //     float s1 = cv::sum(patch1)(0);
    //     float s2 = cv::sum(patch2)(0);
    //     float prod = patch1.dot(patch2);
    //     float s = prod - s1 * mean2(0) - s2 * mean1(0) + N * mean1(0) * mean2(0);
    //     ncc = s / (sigma1(0) * sigma2(0)) / (N);
    // }
    // else
    // {
    //     ncc = -1.0f;
    // }

    // return ncc;

    cv::Mat ncc;
    cv::matchTemplate(patch1, patch2, ncc, cv::TM_CCOEFF_NORMED);

    return ncc.at<float>(0, 0);
}


/**
 * Intersection over Union
 */
float tld::utils::IoU(const BBox& bbox1, const BBox& bbox2)
{
    float I = (bbox1 & bbox2).area();  // intersection
    float U = bbox1.area() + bbox2.area() - I;  // union
    return I / U;
}


/**
 * Non-maximal suppression.
 * Clusters neighboring bboxes and and for each clusters computes the average bbox.
 */
std::vector<BBox> tld::utils::NMS(const std::vector<BBox> &bboxSet,
                                            float overlapThreshold)
{
    if (bboxSet.empty())
    {
        return std::vector<BBox>();
    }

    // Calculate pairwise overlaps between all the bboxes in the bboxSet and create adjecency list
    int N = bboxSet.size();
    std::vector<std::vector<int>> adjacencyList(N);
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float iou = tld::utils::IoU(bboxSet[i], bboxSet[j]);
            if (iou >= overlapThreshold)
            {
                adjacencyList[i].push_back(j);
            }
        }
    }

    // Cluster the adjacent bboxes using BFS
    std::vector<std::vector<int>> clusters;
    int clusterIndex = 0;
    std::queue<int> queue;
    std::set<int> seenIndices;
    for (int i = 0; i < N; ++i)
    {
        std::vector<int> bboxIndices;
        if (seenIndices.empty() || seenIndices.find(i) == seenIndices.end())
        {
            seenIndices.insert(i);
            bboxIndices.push_back(i);
            queue.push(i);
            while(!queue.empty())
            {
                int ii = queue.front();
                queue.pop();
                for (const int& idj : adjacencyList[ii])
                {
                    if (seenIndices.find(idj) == seenIndices.end())
                    {
                        seenIndices.insert(idj);
                        bboxIndices.push_back(idj);
                        queue.push(idj);
                    }
                }
            }
            clusters.push_back(bboxIndices);
            clusterIndex++;
        }
    }

    // Compute the average bbox for each cluster
    std::vector<BBox> avgBboxes;
    for (const auto& cluster : clusters)
    {
        int clusterSize = cluster.size();
        float x = 0.0f, y = 0.0f;
        float w = 0.0f, h = 0.0f;
        for (const int& i : cluster)
        {
            BBox bbox = bboxSet[i];
            x += bbox.x;
            y += bbox.y;
            w += bbox.width;
            h += bbox.height;
        }
        BBox avgBbox(x / clusterSize, y / clusterSize, w / clusterSize, h / clusterSize);
        avgBboxes.push_back(avgBbox);
    }

    return avgBboxes;
}


/**
 * Computes the integral image of both the input image and its squares
 * and stores them to iImage and iImageSq respectively.
 */
void tld::utils::computeIntegralImage2(const cv::Mat& image, cv::Mat& iImage, cv::Mat& iImageSq)
{
    CV_Assert(image.type() == CV_8UC1);

    const int nRows = image.rows;
    const int nCols = image.cols;

    CV_Assert(255.0 * nRows * nCols < std::numeric_limits<int>::max());
    CV_Assert(255.0 * 255.0 * nRows * nCols < std::numeric_limits<float>::max());

    iImage = cv::Mat::zeros(nRows + 1, nCols + 1, CV_32SC1);
    iImageSq = cv::Mat::zeros(nRows + 1, nCols + 1, CV_32FC1);
    
    for (int i = 1; i <= nRows; ++i)
    {
        // Get the pointer to the ith and (i-1)th rows of the iImage
        int* const iImageRowPtr = iImage.ptr<int>(i);
        const int* const iImagePrevRowPtr = iImage.ptr<int>(i - 1);
        // Get the pointer to the ith and (i-1)th rows of the iImageSq
        float* const iImageSqRowPtr = iImageSq.ptr<float>(i);
        const float* const iImageSqPrevRowPtr = iImageSq.ptr<float>(i - 1);
        // Get the pointer to the (i-1)th rows of the image
        const uchar* const imagePrevRowPtr = image.ptr<uchar>(i - 1);
        // Compute the integral image and the integral image of squares
        for (int j = 1; j <= nCols; ++j)
        {
            int imgVal = static_cast<int>(imagePrevRowPtr[j - 1]);
            iImageRowPtr[j] = iImageRowPtr[j - 1] + iImagePrevRowPtr[j] - iImagePrevRowPtr[j - 1] + imgVal;
            iImageSqRowPtr[j] = iImageSqRowPtr[j - 1] + iImageSqPrevRowPtr[j] - iImageSqPrevRowPtr[j - 1] + 1.0f * imgVal * imgVal;
        }
    }
}


/**
 * Sum the pixels of an image patch defined by the given bbox using the integral image iImage.
 */
float tld::utils::sumPatch(const cv::Mat& iImage, const BBox& bbox)
{
    float x = bbox.x, y = bbox.y, width = bbox.width, height = bbox.height;
    CV_Assert(x >= 0 && (x + width) <= iImage.cols && y >= 0 && (y + height) <= iImage.rows);

    float A = iImage.at<float>(y, x);  //at(row, col)
    float B = iImage.at<float>(y, x + width);
    float C = iImage.at<float>(y + height, x);
    float D = iImage.at<float>(y + height, x + width);

    return D + A - B - C;
}


/**
 * Opens a text file containing bbox specifications (x, y, width, height)
 * and creates a bbox based on the specifications of the i-th line.
 */
BBox tld::utils::bboxFromFile(const std::string& filename, int lineIndex)
{
    assert(lineIndex > 0);

    std::ifstream input(filename);
    std::string line;
    
    for (int i = 0; i < lineIndex; ++i)
    {
        std::getline(input, line);
    }

    std::stringstream ss(line);

    std::vector<int> bboxSpecs;
    for (int i; ss >> i;)
    {
        bboxSpecs.push_back(i);    
        if (ss.peek() == ',' || ss.peek() == ' ')
        {
            ss.ignore();
        }
    }
    
    return BBox(bboxSpecs[0], bboxSpecs[1], bboxSpecs[2], bboxSpecs[3]);
}




std::string tld::utils::to_string(float x, unsigned n)
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << x;
    return out.str();
}