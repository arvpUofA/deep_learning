#include <fstream>
#include <utility>
#include <vector>
#include <iostream>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv.hpp>

#include <time.h>

#include "utils.h"

using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;

struct Roi {
    cv::Rect rect;
    int tagIdx;
};

int main(int argc, char* argv[]) {

    // Set dirs variables
    string ROOTDIR = "../";
    string LABELS = "demo/yolov3-tiny/labels_map.pbtxt";
    string GRAPH = "demo/yolov3-tiny/frozen_graph.pb";

    // Set input & output nodes names
    string inputLayer = "Placeholder:0";
    std::vector<string> outputLayer = {"detection_boxes:0"};

    // Load and initialize the model from .pb file
    std::unique_ptr<tensorflow::Session> session;
    string graphPath = tensorflow::io::JoinPath(ROOTDIR, GRAPH);
    LOG(INFO) << "graphPath:" << graphPath;
    Status loadGraphStatus = loadGraph(graphPath, &session);
    if (!loadGraphStatus.ok()) {
        LOG(ERROR) << "loadGraph(): ERROR" << loadGraphStatus;
        return -1;
    } else {
        LOG(INFO) << "loadGraph(): frozen graph loaded" << std::endl;
    }

    // Load labels map from .pbtxt file
    std::map<int, std::string> labelsMap = std::map<int,std::string>();
    Status readLabelsMapStatus = readLabelsMapFile(tensorflow::io::JoinPath(ROOTDIR, LABELS), labelsMap);
    if (!readLabelsMapStatus.ok()) {
        LOG(ERROR) << "readLabelsMapFile(): ERROR" << loadGraphStatus;
        return -1;
    } else
        LOG(INFO) << "readLabelsMapFile(): labels map loaded with " << labelsMap.size() << " label(s)" << std::endl;

    cv::Mat frame, inputFrame, resizedCopy;
    Tensor tensor;
    std::vector<Tensor> outputs;
    float thresholdScore = 0.25f;
    float thresholdIOU = 0.8f;

    // FPS count
    int nFrames = 25;
    int iFrame = 0;
    double fps = 0.;
    time_t start, end;
    time(&start);

    // Start streaming frames from camera
    cv::VideoCapture cap("/home/jkc1/arvp/videos/front-cam-dice-roulette.mp4");
    int width = 416;
    int height = 416;
    tensorflow::TensorShape shape({1, height, width, 3});

    cv::Point2d ratio( 
        cap.get(cv::CAP_PROP_FRAME_WIDTH) / width,
        cap.get(cv::CAP_PROP_FRAME_HEIGHT) / height
    );

    std::vector<Roi> rois;
    while (cap.isOpened()) {
        auto t1 = std::chrono::high_resolution_clock::now();

        cap >> frame;
        cv::resize(frame, resizedCopy, cv::Size(height, width));
        cv::cvtColor(resizedCopy, inputFrame, cv::COLOR_BGR2RGB);
        std::cout << "Frame # " << iFrame << std::endl;

        if (nFrames % (iFrame + 1) == 0) {
            time(&end);
            fps = 1. * nFrames / difftime(end, start);
            time(&start);
        }
        iFrame++;

        // Convert mat to tensor
        tensor = Tensor(tensorflow::DT_FLOAT, shape);
        float *p = tensor.flat<float>().data();
        cv::Mat cameraImg(inputFrame.rows, inputFrame.cols, CV_32FC3, p);
        inputFrame.convertTo(cameraImg, CV_32FC3);

        // Run the graph on tensor
        outputs.clear();
        Status runStatus = session->Run({{inputLayer, tensor}}, outputLayer, {}, &outputs);
        if (!runStatus.ok()) {
            LOG(ERROR) << "Running model failed: " << runStatus;
            return -1;
        }

        // Extract results from the outputs vector
        // tensorflow::TTypes<float>::Flat numDetections = outputs[3].flat<float>();
        tensorflow::TTypes<float, 2>::Tensor boxes = outputs[0].flat_inner_dims<float, 2>();
        int predictionCount = boxes.dimension(0);        
        
        int outputSize = boxes.dimension(1); // 9 for 4 classes
        int classCount = outputSize - 5; // 5 is the box attributes + threshold
        for (int i = 0; i < predictionCount; ++i) {
            if (boxes(i, 4) > thresholdScore) {
                std::vector<float> classes(classCount);
                for (int j = 5; j < outputSize; ++j) {
                    classes[j - 5] = (boxes(i, j));
                }

                int tagIdx = std::distance(
                    classes.begin(),
                    std::max_element(classes.begin(), classes.end())
                );

                rois.push_back({
                    cv::Rect(
                        (int) (boxes(i, 0) * ratio.x),
                        (int) (boxes(i, 1) * ratio.y),
                        (int) ((boxes(i, 2) - boxes(i, 0)) * ratio.x),
                        (int) ((boxes(i, 3) - boxes(i, 1)) * ratio.y)
                    ),
                    tagIdx
                });
            }
        }

        int fontCoeff = 12;
        for (Roi roi : rois) {
            cv::rectangle(frame, roi.rect, cv::Scalar(0, 255, 255));

            cv::Point textCorner = cv::Point(roi.rect.x, roi.rect.y + fontCoeff * 0.9);
            cv::putText(frame, labelsMap[roi.tagIdx], textCorner, cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 0, 0));
        }
        
        if (frame.cols <= 0 || frame.rows <= 0) { break; }
        cv::imshow("stream", frame);
        if(cv::waitKey(5) == 27) { break; } // Wait for 'esc' key press to exit
        
        auto t2 = std::chrono::high_resolution_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1); 
        double ms = (1000*diff.count());
        
        std::cout << rois.size() << " detections. Frame time: " << ms << "ms" << std::endl;
        rois.clear();

    }
    cv::destroyAllWindows();

    return 0;
}
