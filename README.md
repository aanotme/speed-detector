# Car Speed Detection (C++ / OpenCV)

This project detects cars in video streams or a live camera feed and
estimates their speed using OpenCV.\
It includes:

-   Haar cascade car detection\
-   Non-maximum suppression (NMS) to avoid multiple boxes per car\
-   Car tracking with smoothed bounding boxes\
-   Speed estimation using pixel-to-meter calibration\
-   Works with video files or iPhone webcam (Continuity Camera) on macOS

------------------------------------------------------------------------

## Installation & Requirements (MacOS)

### Install OpenCV

``` bash
brew install opencv
```

Verify installation:

``` bash
pkg-config --cflags --libs opencv4
```

------------------------------------------------------------------------

## Compile the Program

``` bash
g++ main.cpp -o car_speed `pkg-config --cflags --libs opencv4`
```

------------------------------------------------------------------------

## Running the Program

### 1. Run with a video file

``` bash
./car_speed vid4.mp4
```

### 2. Run with a webcam (Mac / iPhone)

``` bash
./car_speed cam
./car_speed cam 1
```

------------------------------------------------------------------------

## Calibration

Adjust:

``` cpp
double metersPerPixel = 0.05;
```

Use:

    metersPerPixel = real_world_meters / pixel_width

------------------------------------------------------------------------

## Quit

Press ESC, q, or Q.

------------------------------------------------------------------------

## Troubleshooting

-   Try different camera index if webcam fails.
-   Recalibrate metersPerPixel if speed is wrong.
-   Ensure `cars.xml` is in the executable folder.
