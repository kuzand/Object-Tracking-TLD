# Object-Tracking-TLD
C++ implementation of the Tracking-Learning-Detection (TLD) framework for long-term, single-object tracking in a video stream, developed by Kalal et al [1].
As its name suggests, it has three main components: tracking, detection and learning. The tracker estimates object's motion from frame to frame. The detector localizes the object in each frame and if necessary re-initializes the tracker. The results of the tracking and detection are fused into a single result (a bounding box), and if the result is valid then the learning step is performed to estimate the detector's errors and update the dector.
<p align="center">
  <img align="center" width="578" height="585" src="https://user-images.githubusercontent.com/15230238/167124219-2f1e9ca9-6938-4229-ac2b-b7d348bd2c1b.png">
</p>


## Dependencies
* C++17
* OpenCV (>=4.2.0)

## Usage

To build the project with cmake:
```
mkdir build
cd build
cmake ..
make
```
To run it (within the `build/` directory):
```
./my_tld [--input] [--output] [--gt_bboxes] [--evaluate]
```
Options:
* `--input` string, input video path (or keyword "camera").
* `--output` string, output video path (if not specified then no output is produces).
* `--gt_bboxes` string, path to the file containing ground-truth bounding boxes.
* `--evaluate` bool (1 or 0), whether to perform evaluation of the tracking or not (`gt_bboxes` has to be provided).

Examples:
```
--input="../videos/input_video.mp4"
```
```
--input="../Dudek/img/%04d.jpg" --output="../output_video.mp4"  --gt_bboxes="../Dudek/groundtruth_rect.txt" --evaluate=1
```
```
--input="cam" --output="../output_video.mp4"
```
For evaluation we used the tracking benchmark dataset: http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html


## Results

https://user-images.githubusercontent.com/15230238/143576541-51129d7e-5d0c-43d0-a240-9b7503c4406c.mp4

Red bbox is the result of the cascade detector (after NMS), purple bbox of the median-flow tracker, green bbox of the fusion of the former two.

## References
1. [Tracking-Learning-Detection](http://vision.stanford.edu/teaching/cs231b_spring1415/papers/Kalal-PAMI.pdf), Z. Kalal, K. Mikolajczyk, J. Matas.
2. [Robust Object Tracking Based on Tracking-Learning-Detection](https://cvl.tuwien.ac.at/wp-content/uploads/2015/12/thesis.pdf), G. Nebehay.

### Other imlementations
* https://github.com/zk00006/OpenTLD
* https://github.com/gnebehay/OpenTLD
* https://github.com/benpryke/BPTLD
