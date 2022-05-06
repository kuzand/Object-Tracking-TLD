#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <random>


using BBox = cv::Rect2f;

namespace tld
{
	namespace utils
	{
		class Random
		{
		private:
			std::mt19937 engine;

		public:
			unsigned int seed;
			Random() = default;
			Random(unsigned int seed);
			float randf(float low, float high);
			int randi(int low, int high);
			float randN(float low, float high);
		};

		float round(float x, unsigned int n);

		float median(std::vector<float> values);

		void computeIntegralImage2(const cv::Mat &img, cv::Mat &iImage, cv::Mat &iImageSq);

		bool bboxWithinImage(const BBox &bbox, const cv::Mat &image);

		float sumPatch(const cv::Mat &integralImage, const BBox &bbox);

		cv::Mat getPatch(const cv::Mat &image, cv::Point2f patchCenter, cv::Size patchSize);

		float computeNCC(const cv::Mat &patch1, const cv::Mat &patch2);

		float IoU(const BBox &bbox1, const BBox &bbox2); // Intersection over Union

		std::vector<BBox> NMS(const std::vector<BBox> &bboxSet, float overlapThreshold); // Non-Maximal Suppression

		BBox bboxFromFile(const std::string& filename, int lineIndex);

		std::string to_string(float x, unsigned n);

	} // namespace utils
} // namespace tld