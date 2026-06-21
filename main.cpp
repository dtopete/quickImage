#include <iostream>
#include <cstring>
#include <cuda_runtime.h>
#include <cmath>
#include <opencv2/opencv.hpp>

// STB Image headers for image I/O
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define BLOCK_SIZE 256

// Forward declarations of CUDA functions
void cudaAdjustPixels(unsigned char* d_imageData, int width, int height, int channels,
                       float brightness, float contrast, float gamma,
                       float hue, float saturation, float vibrance,
                       bool grayscale);

void cudaApplyFilters(unsigned char* d_imageData, int width, int height, int channels, const char* filterType);

void cudaChromaKey(const unsigned char* d_inputData, unsigned char* d_outputData, 
                   int width, int height, int inChannels, 
                   float targetHue, float hueTolerance, 
                   float minSaturation, float minBrightness);

void cudaApplyMask(const unsigned char* d_inputData, const unsigned char* d_maskData,unsigned char* d_outputData, int width, int height, int inChannels);

bool generateBackgroundMask(const char* inputPath, unsigned char* h_maskData, int width, int height);

// Function to print the usage instructions
void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " <inputImage> [options]\n"
              << "Options:\n"
              << "  -h, --help             Show this help message and exit\n"
              << "  --brightness <value>   Adjust brightness (range: -1.0 to 1.0)\n"
              << "  --contrast <value>     Adjust contrast (range: -1.0 to 1.0)\n"
              << "  --gamma <value>        Adjust gamma (range: 0.1 to 10.0)\n"
              << "  --grayscale            Convert image to grayscale\n"
              << " --saturation <value>    Adjust saturation (range: -1.0 to 1.0)\n"
              << " --hue <value>           Adjust hue (range: -180 to 180 degrees)\n"
              << " --vibrance <value>      Adjust vibrance (range: -1.0 to 1.0)\n" 
              << " --filter <type>         Apply filter (blur, sharpen, vignette)\n"
              << " --chroma                Chroma key filter that removes a given color (green is currently hardcoded)\n"
              << " --removebg              Uses OpenCV to remove background from subject\n"

              << "Future options: Not yet developed \n"
              << " --resize <width> <height> Resize image to specified dimensions\n"
              << " --crop <x> <y> <width> <height> Crop image to specified rectangle\n"
              << " --rotate <angle>        Rotate image by specified angle (degrees)\n"
              << " --flip <direction>      Flip image (horizontal or vertical)\n"
              << " --color-balance <r> <g> <b> Adjust color balance for red, green, blue channels\n"
              << "Another future option:\n"
              << "Batch image processing: Multiple images can receive the same adjustments in one command.\n"
              << "  --batch <file>         Process multiple images listed in a text file (one image path per line)\n"
              << "Advanced features:\n"
              << "LUTs (Look-Up Tables): Apply LUTs for color grading and creative effects.\n"
              << "Super advanced: But video support"
              << "  -o, --output <file>     Output file path (default: output.png)\n"
              << "\nExample:\n"
              << "  " << programName << " input.jpg --brightness 0.2 --contrast 0.1 -o output.png\n";

}

int main(int argc, char* argv[]) {
    // Error if no image was specified
    if (argc < 2) {
        std::cerr << "Error: No input image specified.\n";
        printUsage(argv[0]);
        return 1;
    }

    // Parse command line arguments
    const char* inputPath = argv[1];
    const char* outputPath = "output.png";
    const char* filterType = nullptr;
    float brightness = 0.0f;
    float contrast = 0.0f;
    float gamma = 1.0f;
    float hue = 0.0f;
    float saturation = 0.0f;
    float vibrance = 0.0f;
    bool grayscale = false;
    bool chroma = false;
    bool removeBg = false;
    unsigned char* d_deviceData = nullptr;

    // Grabbing input from user
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--brightness") == 0) {
            if (i + 1 < argc) {
                brightness = std::stof(argv[++i]);
            } else {
                std::cerr << "Error: --brightness requires an argument.\n";
                return 1;
            }
        } else if (strcmp(argv[i], "--contrast") == 0) {
            if (i + 1 < argc) {
                contrast = std::stof(argv[++i]);
            } else {
                std::cerr << "Error: --contrast requires an argument.\n";
                return 1;
            }
        } else if (strcmp(argv[i], "--gamma") == 0) {
            if (i + 1 < argc) {
                gamma = std::stof(argv[++i]);
                if (gamma <= 0.0f) {
                    std::cerr << "Error: --gamma must be greater than 0.\n";
                    return 1;
                }
            } else {
                std::cerr << "Error: --gamma requires an argument.\n";
                return 1;
            }
        } else if (strcmp(argv[i], "--hue") == 0) {
            if (i + 1 < argc) {
                hue = std::stof(argv[++i]);
            } else {
                std::cerr << "Error: --hue requires an argument.\n";
                return 1;
            }
        } else if (strcmp(argv[i], "--saturation") == 0) {
            if (i + 1 < argc) {
                saturation = std::stof(argv[++i]);
            } else {
                std::cerr << "Error: --saturation requires an argument.\n";
                return 1;
            }
        } else if (strcmp(argv[i], "--vibrance") == 0) {
            if (i + 1 < argc) {
                vibrance = std::stof(argv[++i]);
            } else {
                std::cerr << "Error: --vibrance requires an argument.\n";
                return 1;
            }
        } else if (strcmp(argv[i], "--grayscale") == 0) {
            grayscale = true;
        } else if (strcmp(argv[i], "--chroma") == 0) {
            chroma = true;
        } else if (strcmp(argv[i], "--removebg") == 0) {
            removeBg = true;
        } else if (strcmp(argv[i], "--filter") == 0) {
            if (i + 1 < argc) filterType = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
            if (i + 1 < argc) {
                outputPath = argv[++i];
            } else {
                std::cerr << "Error: -o/--output requires an argument.\n";
                return 1;
            }
        } else {
            std::cerr << "Error: Unknown option '" << argv[i] << "'.\n";
            return 1;
        }
    }

    // Load image using STB Image
    int width, height, channels;
    unsigned char* h_imageData = stbi_load(inputPath, &width, &height, &channels, 0);

    if (!h_imageData) {
        std::cerr << "Error: Failed to load image '" << inputPath << "'.\n";
        return 1;
    }

    std::cout << "Loaded image: " << width << "x" << height << " (" << channels << " channels)\n";
    std::cout << "Brightness: " << brightness << ", Contrast: " << contrast << ", Gamma: " << gamma
              << ", Hue: " << hue << ", Saturation: " << saturation << ", Vibrance: " << vibrance
              << ", Grayscale: " << (grayscale ? "on" : "off") << ", Chroma: " 
              << (chroma ? "on" : "off") <<  ", removebg: " << (removeBg ? "on" : "off") << "\n";

    std::cout << "Allocating GPU memory...\n";
    // Allocate GPU memory
    int imageSize = width * height * channels;
    unsigned char* d_imageData;
    cudaMalloc((void**)&d_imageData, imageSize);

    if (!d_imageData) {
        std::cerr << "Error: Failed to allocate GPU memory.\n";
        stbi_image_free(h_imageData);
        return 1;
    }

    std::cout << "Copying image data to GPU...\n";
    // Copy image data to GPU
    cudaMemcpy(d_imageData, h_imageData, imageSize, cudaMemcpyHostToDevice);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (memcpy to device): " << cudaGetErrorString(err) << "\n";
        cudaFree(d_imageData);
        stbi_image_free(h_imageData);
        return 1;
    }

    // Future processing steps (filters, etc.) would be called here as well
    // cudaApplyFilters(d_imageData, width, height, channels, filters);

    // Runs chroma key removal kernel
    if(chroma){
        std::cout << "Running chroma kernel...\n";
        unsigned char* d_rgbaOutput;
        cudaMalloc((void**)&d_rgbaOutput, width * height * 4);

        // Remove green (Hue = 120), with a tolerance of 20 degrees. 
        // Require at least 30% saturation and 30% brightness to avoid removing dark shadows.
        //void cudaChromaKey(const unsigned char* d_inputData, unsigned char* d_outputData, 
        //           int width, int height, int inChannels, 
        //           float targetHue, float hueTolerance, 
        //           float minSaturation, float minBrightness;
        // NOTE: GREEN IS HARDCODED AS THE CHROMA KEY
        cudaChromaKey(d_imageData, d_rgbaOutput, width, height, channels, 
                      120.0f, 20.0f, 0.3f, 0.3f);

        // Free 3 channel device array
        cudaFree(d_imageData);

        // Point data to 4 channel array
        d_imageData = d_rgbaOutput;

        channels = 4;
        imageSize = width*height*4;

        stbi_image_free(h_imageData);
        h_imageData = (unsigned char*)malloc(imageSize);
    } 
    // Process special spatial filters (if requested)
    else if (filterType != nullptr){
        std::cout << "Applying filter: " << filterType << "..\n";
        cudaApplyFilters(d_imageData, width, height, channels, filterType);
    }
    else if (removeBg){
        std::cout << "Remove background kernel\n";
        // Allocate Host memory for mask
        unsigned char* h_maskData = (unsigned char*)malloc(width * height);
        // Generate mask using OpenCV
        if (generateBackgroundMask(inputPath, h_maskData, width, height)) {

            // Moving mask to GPU
            unsigned char* d_maskData;
            cudaMalloc((void**)&d_maskData, width * height);
            cudaMemcpy(d_maskData, h_maskData, width * height, cudaMemcpyHostToDevice);
    
            // 4 channel output array
            unsigned char* d_rgbaOutput;
            cudaMalloc((void**)&d_rgbaOutput, width * height * 4);
    
            // Run Kernel
            cudaApplyMask(d_imageData, d_maskData, d_rgbaOutput, width, height, channels);

            // Free 3 channel device array
            cudaFree(d_imageData);
            d_imageData = d_rgbaOutput;

            channels = 4;
            imageSize = width*height*4;
            stbi_image_free(h_imageData);
            h_imageData = (unsigned char*)malloc(imageSize);
    
            // Cleanup
            cudaFree(d_maskData);
        }
        free(h_maskData);
    }
    else {
        std::cout << "Processing image on GPU...\n";
        // Process basic point color adjustments
        // Apply brightness, contrast, gamma, hue, saturation, and vibrance adjustments
        cudaAdjustPixels(d_imageData, width, height, channels,
                          brightness, contrast, gamma,
                          hue, saturation, vibrance,
                          grayscale);
                          d_deviceData = d_imageData;

    }

    std::cout << "Processing complete. Copying result back to CPU...\n";

    // Copy result back to CPU
    cudaMemcpy(h_imageData, d_imageData, imageSize, cudaMemcpyDeviceToHost);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (memcpy to host): " << cudaGetErrorString(err) << "\n";
        cudaFree(d_imageData);
        stbi_image_free(h_imageData);
        return 1;
    }

    std::cout << "Saving processed image...\n";
    // Save image
    if (stbi_write_png(outputPath, width, height, channels, h_imageData, width * channels)) {
        std::cout << "Successfully saved image to '" << outputPath << "'.\n";
    } else {
        std::cerr << "Error: Failed to save image to '" << outputPath << "'.\n";
        cudaFree(d_imageData);
        stbi_image_free(h_imageData);
        return 1;
    }

    std::cout << "Cleaning up...\n";
    // Cleanup
    cudaFree(d_imageData);
    stbi_image_free(h_imageData);
    std::cout <<  "Processing complete!\n";

    return 0;
}

// 1-channel mask using OpenCV's GrabCut
bool generateBackgroundMask(const char* inputPath, unsigned char* h_maskData, int width, int height) {
    cv::Mat img = cv::imread(inputPath);
    if (img.empty()) {
        std::cerr << "OpenCV failed to load image for GrabCut.\n";
        return false;
    }

    cv::Mat mask;
    cv::Mat bgModel, fgModel;

    // Define bounding box with 10% marging around edges
    cv::Rect rectangle(width * 0.1, height * 0.1, width * 0.8, height * 0.8);

    std::cout << "Running GrabCut algorithm (This takes a while on Bender)\n";
    // Running 5 iterations of GrabCut
    cv::grabCut(img, mask, rectangle, bgModel, fgModel, 5, cv::GC_INIT_WITH_RECT);

    // Convert GrabCut mask into 0 or 255 alpha mask
    for (int i = 0; i < width * height; i++) {
        int y = i / width;
        int x = i % width;
        uchar val = mask.at<uchar>(y, x);
        
        if (val == cv::GC_PR_FGD || val == cv::GC_FGD) {
            h_maskData[i] = 255; // Keep subject
        } else {
            h_maskData[i] = 0;   // Remove background
        }
    }
    return true;
}