[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/N9V_XO6i)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=23771423&assignment_repo_type=AssignmentRepo)

# quickImage - GPU-Accelerated Image Editor By Danny Topete (dtope004)
## Overview
A CUDA-based command-line tool for fast image editing with GPU acceleration. Inspired by the ease of use of `ffmpeg`, quickImage applies point transformations, spatial filters, and background removal using parallel processing.


# [YOUTUBE VIDEO DEMO link](https://youtu.be/LeEC7YI5McE)
- Excuse my sniffles and slow talking, I caught a cold and had difficulty recording this video
- Treat this README as my project report!


## Current Features
- **Basic Adjustments**: Brightness, Contrast, Gamma, Grayscale.
- **Advanced Color Grading**: Hue shifting, Saturation, and Vibrance adjustments utilizing HSV color space conversions.
- **Spatial Filters**: Box Blur, Sharpening, and Vignette effects powered by CUDA convolution kernels.
- **Background Removal**: 
  - Pure CUDA Chroma Keying (Green screen removal).
  - OpenCV-powered AI Subject Segmentation (GrabCut algorithm; which is heavy on bender's CPU).
- **GPU Acceleration**: Uses NVIDIA CUDA for parallel processing.

## Requirements
- NVIDIA GPU with CUDA Compute Capability 3.0 or higher
- CUDA Toolkit 11.0 or later
- OpenCV (Required for GrabCut background removal)
- GCC/G++ compiler
- CMake (version 3.18+)
- `curl` (for setup script)

## Build Instructions

### 1. Download STB Libraries and OpenCV
Run the setup script to download the required image I/O libraries:
```bash
chmod +x setup.sh
./setup.sh
```

The following apptainer holds OpenCV and everyone's libraries
```bash
apptainer shell --nv /scratch/csee147/csee147env.sif
or
apptainer shell --nv /scratch/csee147/csee147env-updated.sif
```

### 2. Build the Project
```bash
cmake .

make
```
To rebuild
```bash
rm -rf CMakeCache.txt CMakeFiles/ Makefile cmake_install.cmake
```

This will create the `quickimage` executable.

## Usage

```bash
./quickimage <inputImage> [options]

Options:
  -h, --help              Show help message and exit
  --brightness <value>    Adjust brightness (range: -1.0 to 1.0)
  --contrast <value>      Adjust contrast (range: -1.0 to 1.0)
  --gamma <value>         Adjust gamma (range: 0.1 to 10.0)
  --grayscale             Convert image to grayscale
  --saturation <value>    Adjust saturation (range: -1.0 to 1.0)
  --hue <value>           Adjust hue (range: -180 to 180 degrees)
  --vibrance <value>      Adjust vibrance (range: -1.0 to 1.0)
  --filter <type>         Apply filter (blur, sharpen, vignette)
  --chroma                Chroma key filter that removes a green background
  --removebg              Uses OpenCV to segment and remove background from subject
  -o, --output <file>     Output file path (default: output.png)
```

### Examples

Increase brightness by 0.2:
```bash
./quickimage input.jpg --brightness 0.2 -o output.png
```

Increase contrast by 0.3:
```bash
./quickimage input.jpg --contrast 0.3 -o output.png
```

Combine brightness and contrast adjustments:
```bash
./quickimage input.jpg --brightness 0.2 --contrast 0.1 -o output.png
```

Cinematic Color grading
```bash
./quickimage portrait.jpg --saturation -0.3 --hue 15 --gamma 1.2 -o cinematic.png
```

Spatial filters
Using blur/sharpen/vignette
```bash
./quickImage input.jpg --filter blur -o blurred.png
./quickImage input.jpg --filter sharpen -o sharp.png
./quickImage input.jpg --filter vignette -o vignette.png
```

Background removal (replaces background for transparent PNG)
```bash
# For green screen photos (Pure CUDA)
# Currently hardcoded to green, can be changed to any chroma color key
./quickimage greenscreen.jpg --chroma -o transparent_subject.png

# For complex backgrounds (OpenCV GrabCut + CUDA)
./quickimage portrait.jpg --removebg -o isolated_subject.png
```


## Implementation Details

### CUDA Kernel (`kernel.cu`)
- **adjustBrightnessContrast**: Parallelized pixel-level adjustment
- Each thread processes one color channel of one pixel
- Block size: 256 threads for optimal GPU utilization

### Image Processing (`main.cpp`)
- Uses STB Image Library for loading/saving PNG, JPG, BMP, and TGA formats
- Supports grayscale and multi-channel images
- Handles memory management between CPU and GPU

1. Per-Pixel Manipulation (adjustPixels)

Unlike simple RGB scaling, true color manipulation requires shifting color spaces.

    - Thread Mapping: Each CUDA thread maps to exactly one pixel across a 1D grid.

    - Algorithm: The thread normalizes the RGB values to [0, 1], applies contrast and brightness, and then converts the RGB values into HSV (Hue, Saturation, Value).

    - In HSV space, the thread modifies the hue angle (0-360 degrees) and scales the saturation/vibrance safely without corrupting the core luminosity. It then converts back to RGB and clamps the data.

2. Spatial Filters (convolutionKernel)

Blur and sharpen filters require a pixel to read the data of its neighbors.

    - Algorithm: The kernel applies a 3x3 convolution matrix over the image. The matrix values are hardcoded and can be increased to make the effect more apperent 
    - The kernel can also be made larger, such as a 5x5 kernel to have larger box blurs

    - To prevent race conditions (where threads overwrite data while neighbors are still reading it), a temporary buffer is allocated on the device (cudaMemcpyDeviceToDevice). The threads read from the buffer and write to the output image.

    - To make the blur and sharpen effects more apperent, I hard coded multiple passes of the kernel through the C++ wrapper function.

3. Pure CUDA Chroma Key (chromaKeyKernel) background removal

    - The kernel converts each pixel to HSV (Hue, Saturation, Value) color space.

    - It calculates the absolute distance between the pixel's hue and the target background hue (120 degrees is hardcoded for green; hue = 120).

    - If the pixel matches the hue within a specific tolerance (hardcoded to 30), and meets minimum brightness/saturation (hardcoded to 0.3f) thresholds, its Alpha channel is forced to 0 (transparent). Otherwise, it is forced to 255 (opaque).

4. Hybrid OpenCV Background Removal (--removebg)

For complex background removals, quickImage uses a CPU/GPU approach to background removal:

    - CPU: OpenCV's GrabCut algorithm processes the image, assuming the subject occupies the center 80% of the frame. It generates a 1-channel alpha mask. It runs slowly on Bender, takes around a minute, but my 12 core Ryzen 9 9900x runs it within a second.

    - The generated mask sent to the GPU including 0 (transparent) or 255 (opaque).

    - GPU: applyMaskKernel copies the original RGB channels and sets adds a 4th (Alpha) channel directly from the OpenCV mask, converts the image into an RGBA format to export the image as a PNG.

## GPU Acceleration and Parallel Execution Details
1. Parallelizing software Pipeline stage
  - CPU does image decoding and loading using stb_image or OpenCV
  - Transfer to the GPU the raw byte array of the image to device memory (cudaMemcpyHostToDevice)
  - GPU Parallelizes the application of filters, colorspace conversion, and convolutions are executed parallel in the GPU
  - Modified byte array is copied back to the Host
  - The Host encodes the image and saves the output to disk

## Cleanup

To remove build artifacts:
```bash
make clean
# or
rm -rf CMakeCache.txt CMakeFiles/ Makefile cmake_install.cmake
```

## Evaluation and Results
- My per-pixel manipulation came out very successful
- My Spatial filters required adding many passes to achieve a noticeable effect for the blur and sharpening. Then the vignette kernel was quite easy.
- The CUDA Chrome key was successful and was great at removing blue screens and green screens (when the hardcoded hue was changed)
- The background removal using OpenCV required GrabCut algorithm, which is CPU based. It would bog down Bender's CPU and the results would be questionable. I do know that it requires more tuning to get a better output. Another factor can be how the pictures decided how good the effect would come out. Also, having the limitation of cli, and not being able to draw a bounding box gave it some limitation to poor bounding boxes.

## Problems Faced
- I had a couple issues with converting the images over to 4 channels and had weird artifacting that took a while to debug
- There is the issue where I had to hardcode values due to time constraints
- The OpenCV's GrabCut Algorithm for background removal is very CPU heavy and bogs down Bender's CPU. To reduce the effect of it, I had to reduce the amount of iterations, but locally, it runs really quickly on a modern 12 core CPU.

## List of Tasks
- Not necessary because this was a solo project

## Project Status
See `projectProposal.typ` for planned features I couldn't get to due to time limitations and debugging:
- LUT-based color grading
- Film simulation filters
- Video support
- Resize/crop/rotate/flip image
- color balance / white balance an image
- Batch image processing