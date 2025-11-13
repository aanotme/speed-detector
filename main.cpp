#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

struct TrackedCar {
    int id;
    cv::Point2f lastPos;
    double lastTime;      // seconds
    double speedKmh;      // latest speed
    cv::Rect bbox;
    cv::Point2f smoothedPos;
    float smoothedWidth;
    float smoothedHeight;
    int missedFrames;
};

int main(int argc, char** argv) {
    // ------------- CONFIGURATION -------------
    // Path to your car cascade file
    std::string cascadePath = "cars.xml"; // TODO: set correctly

    // Approximate conversion from pixels to meters for this scene.
    // Calibrate using lane width in the bottom part of the image:
    // laneWidthMeters / laneWidthPixels.
    double laneWidthMeters = 3.7;      // adjust if your lane width is different
    double laneWidthPixels = 260.0;    // measure this from a frame (in pixels)
    double metersPerPixel = laneWidthMeters / laneWidthPixels;

    // Max distance (in pixels) to associate a detection with an existing track
    const double maxMatchDistance = 80.0;

    // Max missed frames before dropping a track
    const int maxMissedFrames = 10;

    // ----------- OPEN VIDEO / CAMERA ----------
    cv::VideoCapture cap;

    if (argc > 1 && std::string(argv[1]) != "cam") {
        // Argument is a file path or stream URL
        std::string source = argv[1];
        cap.open(source);
        std::cout << "Opening video/stream: " << source << std::endl;
    } else {
        // Default camera (use your iPhone-as-webcam here)
        int camIndex = 0;
        if (argc > 2) {
            camIndex = std::stoi(argv[2]);
        }
        cap.open(camIndex);
        std::cout << "Opening camera index: " << camIndex << std::endl;
    }

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video source." << std::endl;
        return -1;
    }

    // Try to get FPS; if not available, we'll use timer
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0.0) {
        std::cout << "FPS not reported by source, will use real-time timing." << std::endl;
    } else {
        std::cout << "Reported FPS: " << fps << std::endl;
    }

    // ----------- LOAD CASCADE CLASSIFIER -----------
    cv::CascadeClassifier carCascade;
    if (!carCascade.load(cascadePath)) {
        std::cerr << "Error: Could not load cascade file: " << cascadePath << std::endl;
        return -1;
    }

    std::vector<TrackedCar> tracks;
    int nextId = 1;

    // For timing
    int64 tickFreq = cv::getTickFrequency();

    while (true) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            std::cout << "End of video or cannot grab frame." << std::endl;
            break;
        }

        // Current time in seconds (relative to start)
        double nowSec = static_cast<double>(cv::getTickCount()) / tickFreq;

        // ----------- REGION OF INTEREST (bottom part of road) -----------
        int roiTop = static_cast<int>(frame.rows * 0.45); // use lower ~55% of frame
        cv::Rect roi(0, roiTop, frame.cols, frame.rows - roiTop);
        cv::Mat frameRoi = frame(roi);

        // ----------- CAR DETECTION -----------
        cv::Mat gray;
        cv::cvtColor(frameRoi, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);

        std::vector<cv::Rect> detections;
        carCascade.detectMultiScale(
            gray, detections,
            1.05,              // smaller scale step for better multi-size detection
            2,                 // slightly looser to reduce misses
            0,                 // flags
            cv::Size(60, 60)   // min size: cars in ROI are reasonably large
        );

        // ----------- TRACK ASSOCIATION -----------
        // Mark all tracks as missed (will reset below)
        for (auto &t : tracks) {
            t.missedFrames++;
        }

        // For each detection, try to match with existing track
        for (const auto &det : detections) {
            // Convert ROI-relative detection to full-frame coordinates
            cv::Rect detGlobal(det.x,
                               det.y + roiTop,
                               det.width,
                               det.height);
            cv::Point2f center(detGlobal.x + detGlobal.width / 2.0f,
                               detGlobal.y + detGlobal.height / 2.0f);

            int bestIdx = -1;
            double bestDist = maxMatchDistance;

            for (size_t i = 0; i < tracks.size(); ++i) {
                double dx = center.x - tracks[i].lastPos.x;
                double dy = center.y - tracks[i].lastPos.y;
                double dist = std::sqrt(dx * dx + dy * dy);
                if (dist < bestDist) {
                    bestDist = dist;
                    bestIdx = static_cast<int>(i);
                }
            }

            if (bestIdx >= 0) {
                // Match found: update existing track
                TrackedCar &tc = tracks[bestIdx];
                // Smooth position to reduce jitter
                const float posAlpha = 0.3f; // between 0 and 1, lower = smoother
                tc.smoothedPos = (1.0f - posAlpha) * tc.smoothedPos + posAlpha * center;
                // Smooth box size to reduce flickering big/small boxes
                const float sizeAlpha = 0.3f;
                tc.smoothedWidth = (1.0f - sizeAlpha) * tc.smoothedWidth + sizeAlpha * static_cast<float>(det.width);
                tc.smoothedHeight = (1.0f - sizeAlpha) * tc.smoothedHeight + sizeAlpha * static_cast<float>(det.height);
                if (tc.lastTime > 0.0) {
                    double dt = nowSec - tc.lastTime;
                    if (dt > 0.0) {
                        // Only trust speed when car is in the lower part of the frame
                        int speedYMin = static_cast<int>(frame.rows * 0.60);
                        int speedYMax = frame.rows;
                        if (tc.smoothedPos.y > speedYMin && tc.smoothedPos.y < speedYMax) {
                            double pixelDist = std::hypot(tc.smoothedPos.x - tc.lastPos.x,
                                                          tc.smoothedPos.y - tc.lastPos.y);
                            double meters = pixelDist * metersPerPixel;
                            double speedMps = meters / dt;
                            double instantSpeedKmh = speedMps * 3.6;
                            // Exponential moving average for speed to make it more readable
                            const double speedAlpha = 0.2; // between 0 and 1, lower = smoother
                            if (tc.speedKmh == 0.0) {
                                tc.speedKmh = instantSpeedKmh;
                            } else {
                                tc.speedKmh = speedAlpha * instantSpeedKmh + (1.0 - speedAlpha) * tc.speedKmh;
                            }
                        }
                    }
                }
                tc.lastPos = tc.smoothedPos;
                tc.lastTime = nowSec;
                tc.bbox = detGlobal;
                tc.missedFrames = 0;
            } else {
                // No match: create new track
                TrackedCar tc;
                tc.id = nextId++;
                tc.lastPos = center;
                tc.smoothedPos = center;
                tc.lastTime = nowSec;
                tc.speedKmh = 0.0;
                tc.bbox = detGlobal;
                tc.smoothedWidth = static_cast<float>(det.width);
                tc.smoothedHeight = static_cast<float>(det.height);
                tc.missedFrames = 0;
                tracks.push_back(tc);
            }
        }

        // Remove tracks that have been missed too long
        tracks.erase(
            std::remove_if(tracks.begin(), tracks.end(),
                           [maxMissedFrames](const TrackedCar &tc) {
                               return tc.missedFrames > maxMissedFrames;
                           }),
            tracks.end()
        );

        // ----------- DRAW RESULTS -----------
        for (const auto &tc : tracks) {
            cv::Rect bbox;
            bbox.width = static_cast<int>(tc.smoothedWidth);
            bbox.height = static_cast<int>(tc.smoothedHeight);
            // Re-center bbox around smoothed position to reduce jitter in drawing
            bbox.x = static_cast<int>(tc.smoothedPos.x - bbox.width / 2.0f);
            bbox.y = static_cast<int>(tc.smoothedPos.y - bbox.height / 2.0f);
            // Draw bounding box
            cv::rectangle(frame, bbox, cv::Scalar(0, 255, 0), 2);

            // Prepare label text: ID + speed
            std::ostringstream oss;
            oss << "ID " << tc.id << " | "
                << std::fixed << std::setprecision(1)
                << tc.speedKmh << " km/h";

            std::string label = oss.str();

            int baseLine = 0;
            cv::Size labelSize = cv::getTextSize(
                label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine
            );
            int top = std::max(bbox.y, labelSize.height);

            cv::rectangle(
                frame,
                cv::Point(bbox.x, top - labelSize.height - 5),
                cv::Point(bbox.x + labelSize.width, top + baseLine),
                cv::Scalar(0, 0, 0),
                cv::FILLED
            );

            cv::putText(
                frame, label,
                cv::Point(bbox.x, top - 2),
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                cv::Scalar(255, 255, 255),
                1
            );
        }

        cv::imshow("Car Detection & Speed Estimation", frame);

        char key = static_cast<char>(cv::waitKey(1));
        if (key == 27 || key == 'q' || key == 'Q') { // ESC or q to quit
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}