#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <deque>
#include "Utils.h"
#include "TLD.h"


static void readme()
{
    std::cout << "Please provide the following arguments:\n"
              << "--input : string, input video path (or \"camera\" keyword).\n"
              << "--output : string, output video path (if now specified then no output file will be produces).\n"
              << "--gt_bboxes : string, path to the file containing ground truth bounding boxes.\n"
              << "--evaluate : bool (1 or 0), whether to perform evaluation of the tracking results or not (gt_bboxes has to be provided)."
              << std::endl;
}


static const cv::String args = "{input||input video path}"
                               "{output||output video path}"
                               "{gt_bboxes||ground truth bboxes path}"
                               "{evaluate||evaluate the tracking (only if the ground truth bboxes are provided)}";


int main(int argc, char* argv[])
{   
    // Parse arguments
    //--input="../benchmark/Dudek/img/%2504d.jpg"  --gt_bboxes="../benchmark/Dudek/groundtruth_rect.txt" --evaluate=1
    // 
    std::string inputPath;
    std::string outputPath;
    std::string gtBboxesPath;
    bool evaluate = false;
    cv::CommandLineParser parser(argc, argv, args);
    if (parser.has("input"))
        inputPath = parser.get<cv::String>("input");
    if (parser.has("output"))
        outputPath = parser.get<cv::String>("output");
    if (parser.has("gt_bboxes"))
        gtBboxesPath = parser.get<cv::String>("gt_bboxes");
    if (parser.has("gt_bboxes") && parser.has("evaluate"))
    {
        evaluate = parser.get<bool>("evaluate");
    }

    // Create a VideoCapture object and open the input file
    cv::VideoCapture inputVideo;
    if (inputPath.empty())
    {
        readme();
        return 1;
    }
    else if (inputPath == "camera")
    {
        inputVideo.open(0);
    }
    else
    {
        inputVideo.open(inputPath);
    }

    // Check if camera was opened successfully
    if(!inputVideo.isOpened())
    {
        std::cout << "Error opening video stream or file" << std::endl;
        return 1;
    }

    // Read the first frame
    cv::Mat initialFrame;
    cv::Mat initialFrameGray;
    inputVideo.read(initialFrame);
    cv::cvtColor(initialFrame, initialFrameGray, cv::COLOR_BGR2GRAY);

    int totalFrames = inputVideo.get(cv::CAP_PROP_FRAME_COUNT);     
    int videoWidth = inputVideo.get(cv::CAP_PROP_FRAME_WIDTH);
    int videoHeight = inputVideo.get(cv::CAP_PROP_FRAME_HEIGHT);

    // Create a VideoWriter object
    float fpsMax = 25.0f;
    cv::VideoWriter outputVideo;
    if (!outputPath.empty())
    {
        outputVideo.open(outputPath,
                    cv::VideoWriter::fourcc('M','J','P','G'),
                    fpsMax,
                    cv::Size(videoWidth, videoHeight));
    }

	// Select the initial bbox enclosing the object of interest
	BBox initialBbox;
    if (gtBboxesPath.empty())
    {
		initialBbox = cv::selectROI(initialFrame, false);
		cv::destroyWindow("ROI selector");
	}
	else
	{
		initialBbox = tld::utils::bboxFromFile(gtBboxesPath, 1);
	}

    std::cout << "Initial bbox: ("
              << initialBbox.x << ", "
              << initialBbox.y << ", "
              << initialBbox.width << ", "
              << initialBbox.height << ")"
              << std::endl;

    // ------------------
    // Initialize the TLD
    // ------------------
    tld::TLD myTLD(initialFrameGray, initialBbox);

    // Main video loop
    cv::Mat newFrame;
    float avgFPS = 0.0f;
    int frameCounter = 1;
    bool pausedVideo = false;
    int TP = 0;  // for evaluation
    int FN = 0;  // for evaluation
    float tau = 0.50;  // for evaluation
    while(inputVideo.read(newFrame) && !pausedVideo)
    {
        // Start the timer
        double timer = double(cv::getTickCount());

        frameCounter++;  

        // Convert to gray scale
        cv::Mat newFrameGray;
        cv::cvtColor(newFrame, newFrameGray, cv::COLOR_BGR2GRAY);

        // RUN TLD
        BBox trackedBbox;
        std::vector<BBox> detectedBboxes;
        BBox fusedBbox;
        myTLD.run(newFrameGray, trackedBbox, detectedBboxes, fusedBbox);

         // Draw the tracked bbox (purple color)
         if (!trackedBbox.empty())
         {  
             cv::rectangle(newFrame, trackedBbox, cv::Scalar( 229, 55, 148 ), 5, 1 );
         }
         // Draw the detected bboxes (red color)
         if (!detectedBboxes.empty())
         {
             for (const BBox& detectedBbox : detectedBboxes)
             {
                 cv::rectangle(newFrame, detectedBbox, cv::Scalar( 0, 0, 255 ), 4, 1 );
             }
         }

        // Draw the fused bbox (green color)
        if (!fusedBbox.empty())
        {
            cv::rectangle(newFrame, fusedBbox, cv::Scalar( 0, 255, 128 ), 2, 1 );
        }


        // Get the ground truth bbox and compare with the fused bbox
        if (evaluate)
        {
            BBox gtBbox = tld::utils::bboxFromFile(gtBboxesPath, frameCounter);

            float overlap = tld::utils::IoU(fusedBbox, gtBbox);
            if (overlap > tau)
            {
                TP++;
            }
            else
            {
                FN++;
            }

            // Draw the gt bbox (blue color)
            if (!gtBbox.empty())
            {  
                cv::rectangle(newFrame, gtBbox, cv::Scalar( 255, 0, 0 ), 2, 1 );
            }
        }


         // Draw the object model positive and negative templates
         for (std::size_t i = 0; i < myTLD.objectModel.positiveTemplates.size(); ++i)
         {
             cv::Mat posTemplate;
             cv::Mat posTemplateResized;
             cv::cvtColor(myTLD.objectModel.positiveTemplates[i], posTemplate, cv::COLOR_GRAY2RGB);
             cv::resize(posTemplate, posTemplateResized, cv::Size(newFrame.cols / myTLD.params.MAX_OBJ_MODEL_SIZE, newFrame.cols / myTLD.params.MAX_OBJ_MODEL_SIZE), 0, 0, cv::INTER_CUBIC);
             posTemplateResized.copyTo(newFrame(cv::Rect2f(i * posTemplateResized.cols, 0.0f, posTemplateResized.cols, posTemplateResized.rows)));
         }

         for (std::size_t i = 0; i < myTLD.objectModel.negativeTemplates.size(); ++i)
         {
             cv::Mat negTemplate;
             cv::Mat negTemplateResized;
             cv::cvtColor(myTLD.objectModel.negativeTemplates[i], negTemplate, cv::COLOR_GRAY2RGB);
             cv::resize(negTemplate, negTemplateResized, cv::Size(newFrame.cols / myTLD.params.MAX_OBJ_MODEL_SIZE, newFrame.cols / myTLD.params.MAX_OBJ_MODEL_SIZE), 0, 0, cv::INTER_CUBIC);
             negTemplateResized.copyTo(newFrame(cv::Rect2f(i * negTemplateResized.cols, negTemplateResized.rows, negTemplateResized.cols, negTemplateResized.rows)));
         }

        // Calculate the FPS
        float fps = 1.0f * cv::getTickFrequency() / (cv::getTickCount() - timer);
        avgFPS += fps;

        // Display some useful info
        
        if (evaluate)
        {
            cv::putText(newFrame, "Recall: " + tld::utils::to_string(1.0f * TP / (TP + FN), 3),
                cv::Point(10, newFrame.rows - 130),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(127, 0, 255), 2);
        }
        cv::putText(newFrame, "Frame size: " + std::to_string(newFrame.rows) + "x" + std::to_string(newFrame.cols),
            cv::Point(10, newFrame.rows - 100),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(127, 0, 255), 2);

        cv::putText(newFrame, "Frame: " + std::to_string(frameCounter) + "/" + std::to_string(totalFrames),
                    cv::Point(10, newFrame.rows - 70),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(127, 0, 255), 2);

        cv::putText(newFrame, "Subwindows: " + std::to_string(myTLD.detector.ensClfPool.size()),
                    cv::Point(10, newFrame.rows - 40),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(127, 0, 255), 2);

        cv::putText(newFrame, "FPS: " + tld::utils::to_string(fps, 2),
                    cv::Point(10, newFrame.rows - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(127, 0, 255), 2);

        
        // Write the processed frame to a video file
        if (!outputPath.empty())
        {
            outputVideo.write(newFrame);
        }

        // Display the processed frame
        cv::imshow("Tracking", newFrame);

        // Wait for a key to be pressed (Space to pause/play video, Esc to exit)
        enum Key {ESC = 27, SPACE = 32};
        switch (cv::waitKey(int(1000 / fpsMax)))
        {
            case SPACE:
                while (true)
                {
                    int k = cv::waitKey(0);
                    if (k == SPACE)
                    {
                        break;
                    }
                    else if (k == ESC)
                    {
                        pausedVideo = true;
                        break;
                    }
                }
                break;

            case ESC:
                pausedVideo = true;
                break;
        } 
    }

    avgFPS /= frameCounter;
    std::cout << "Average FPS: " << avgFPS << std::endl;

    if (evaluate)
    {
        float recall = 1.0f * TP / (TP + FN);
        std::cout << "Recall: " << recall << std::endl;
    }

    std::cout << "Done!" << std::endl;

    // Release the video capture object
    cv::waitKey(0);
    inputVideo.release();
    if (!outputPath.empty())
    {
        outputVideo.release();
    }        
    cv::destroyAllWindows();

    return 0;
}