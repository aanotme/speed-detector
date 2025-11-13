#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char** argv) {
    int camIndex = 0;
    if (argc > 1) {
        camIndex = std::stoi(argv[1]);
    }

    cv::VideoCapture cap(camIndex, cv::CAP_AVFOUNDATION);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open camera index " << camIndex << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (true) {
        if (!cap.read(frame) || frame.empty()) {
            std::cerr << "Cannot grab frame from camera " << camIndex << std::endl;
            break;
        }

        cv::imshow("Test Camera", frame);
        char key = (char)cv::waitKey(1);
        if (key == 27 || key == 'q') break; // ESC or q
    }

    return 0;
}