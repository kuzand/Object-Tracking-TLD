#pragma once

#include <opencv2/opencv.hpp>
#include "MedianFlowTracker.h"
#include "CascadeClassifier.h"
#include "ObjectModel.h"
#include "Params.h"
#include "Utils.h"


using BBox = cv::Rect2f;

namespace tld
{
class TLD
{
public:

	Params params;
	
	TLD(const cv::Mat &initialFrame,
		const BBox &initialBbox);

	ObjectModel objectModel;
	MedianFlowTracker tracker;
	CascadeClassifier detector;

	void run(const cv::Mat &frame,
			BBox &trackedBbox,
			std::vector<BBox> &detectedBboxes,
			BBox &fusedBbox);


private:

	tld::utils::Random rng;

	bool isValidPrevBbox;
	
	BBox track(const cv::Mat &frame);

	std::vector<BBox> detect(const cv::Mat &frame) const;

	BBox fuse(const cv::Mat &frame,
			  const BBox &trackedBbox,
			  const std::vector<BBox> &detectedBboxes);

	void learn(const cv::Mat &frame, const BBox& fusedBbox);

};

} // namespace tld