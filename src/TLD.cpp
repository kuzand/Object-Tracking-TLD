#include "TLD.h"
#include "Utils.h"


tld::TLD::TLD(const cv::Mat &initialFrame,
              const BBox &initialBbox)
{
    // Initialize parameters
    //this->params = Params();
    //this->params.write("../params.yaml");
    this->params.read("../params.yaml");
    this->params.printParams();

    // Random number generator
    this->rng = tld::utils::Random(params.RNG_SEED);

    this->isValidPrevBbox = false;

    // Initialize the object model
    this->objectModel = ObjectModel(initialFrame, initialBbox, &params, &rng);

    // Initialize the tracker
    this->tracker = MedianFlowTracker(initialFrame, initialBbox, &params, &rng);

    // Initialize the detector
    this->detector = CascadeClassifier(initialFrame, initialBbox, objectModel, &params, &rng);

    // Run the learn method for the initial frame and bbox
    this->learn(initialFrame, initialBbox);
}


void tld::TLD::run(const cv::Mat &frame,
                   BBox &trackedBbox,
                   std::vector<BBox> &detectedBboxes,
                   BBox &fusedBbox)
{
        // TRACKING
        trackedBbox = this->track(frame);

        // DETECTION
        detectedBboxes = this->detect(frame);

        // FUSION
        fusedBbox = this->fuse(frame, trackedBbox, detectedBboxes);

        // LEARNING
        if (this->isValidPrevBbox)
        {
            this->learn(frame, fusedBbox);
        }
}



BBox tld::TLD::track(const cv::Mat &frame)
{
    BBox trackedBbox = tracker.track(frame);

    return trackedBbox;
} 


std::vector<BBox> tld::TLD::detect(const cv::Mat &frame) const
{
    std::vector<BBox> detectedBboxes = detector.detect(frame);

    return detectedBboxes;
}


BBox tld::TLD::fuse(const cv::Mat &frame,
                    const BBox &trackedBbox,
                    const std::vector<BBox> &detectedBboxes)
{
    if (detectedBboxes.empty() && trackedBbox.empty())
    {
        this->isValidPrevBbox = false;
        this->tracker.reinitialize(frame, BBox());
        return BBox();
    }

    BBox fusedBbox;
    bool isValidBbox = false;

    // Confidence of the first detection result
    float pD = 0.0f;
    if (!detectedBboxes.empty())
    {
        cv::Mat detectedPatch = frame(detectedBboxes[0]);
        pD = this->detector.templateMatching(detectedPatch);
    }

    if (!trackedBbox.empty())
    {
        // Confidence of the tracking result
        cv::Mat trackedPatch = frame(trackedBbox);
        float pR = this->detector.templateMatching(trackedPatch);

        if ((detectedBboxes.size() == 1) && (pD > pR))
        {
            fusedBbox = detectedBboxes[0];
            // Re-initialize the tracker
            this->tracker.reinitialize(frame, fusedBbox);
        }
        else
        {
            fusedBbox = trackedBbox;

            if ((pR > params.THETA_PLUS) || (this->isValidPrevBbox && pR > params.THETA_MINUS))
            {
                isValidBbox = true;
            }
        }
    }
    else if (detectedBboxes.size() == 1)
    {
        fusedBbox = detectedBboxes[0];
        // Re-initialize the tracker
        this->tracker.reinitialize(frame, fusedBbox);
    }

    this->isValidPrevBbox = isValidBbox;

    return fusedBbox;
}


// Called only if fusedBbox is valid (which means that the tracking bbox was selected).
void tld::TLD::learn(const cv::Mat &frame, const BBox& fusedBbox)
{
    float pBfused = this->detector.templateMatching(frame(fusedBbox));
    
    for (std::size_t i = 0; i < this->detector.ensClfPool.size(); ++i)
    {
        tld::EnsembleClassifier& ensClf = this->detector.ensClfPool[i];
        BBox bbox = ensClf.bbox;
        float overlap = tld::utils::IoU(bbox, fusedBbox);
        float patchConfidence  = ensClf.classifyPatch(frame);
        // P-expert (bbox is false negative)
        if (overlap > 0.6f && patchConfidence < 0.5f)
        {
            // Update the classifier
            for (int k = 0; k < ensClf.numFerns; ++k)
            {
                int Fk = ensClf.ferns[k].calcFern(frame, bbox);
                ensClf.ferns[k].numPos[Fk] += 1;
            }
        }
        // N-expert (bbox is false positive)
        else if (overlap < 0.2f && patchConfidence > 0.5f)
        {
            // Update the classifier
            for (int k = 0; k < ensClf.numFerns; ++k)
            {
                int Fk = ensClf.ferns[k].calcFern(frame, bbox);
                ensClf.ferns[k].numNeg[Fk] += 1;
            }

            // Check to update the object model
            //if (pBfused > params.getParams().THETA_MINUS)
            {
                cv::Mat negativeTemplate = frame(bbox);
                cv::Mat negativeTemplateResized;
                cv::resize(negativeTemplate, negativeTemplateResized,
                            params.TEMPLATE_SIZE, 0, 0, cv::INTER_CUBIC);
                this->objectModel.addNegativeTemplate(negativeTemplateResized);
            }
        }
    }
    
    if (pBfused < params.THETA_PLUS)  // note that pBfused > THETA_MINUS
    {
        cv::Mat positivePatch = frame(fusedBbox);
        cv::Mat positivePatchResized;
        cv::resize(positivePatch, positivePatchResized,
                    params.TEMPLATE_SIZE, 0, 0, cv::INTER_CUBIC);
        this->objectModel.addPositiveTemplate(positivePatchResized);
    }
}