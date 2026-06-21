#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <cstdio>
#include <cstring>

#define BLOCK_SIZE 256

// Converts RGB to Hue, Saturation, Value color space
// Required for hue, saturation, and vibrance adjustments
// Source: https://en.wikipedia.org/wiki/HSL_and_HSV#From_RGB
__device__ void rgbToHsv(float r, float g, float b, float* h, float* s, float* v) {
    float maxc = fmaxf(r, fmaxf(g, b));
    float minc = fminf(r, fminf(g, b));
    float d = maxc - minc;
    *v = maxc;
    *s = (maxc == 0.0f) ? 0.0f : d / maxc;

    if (d == 0.0f) {
        *h = 0.0f;
        return;
    }

    if (maxc == r) {
        *h = 60.0f * fmodf((g - b) / d, 6.0f);
    } else if (maxc == g) {
        *h = 60.0f * (((b - r) / d) + 2.0f);
    } else {
        *h = 60.0f * (((r - g) / d) + 4.0f);
    }

    if (*h < 0.0f) {
        *h += 360.0f;
    }
}

// Converts Hue, Saturation, Value back to RGB color space
// Required for hue, saturation, and vibrance adjustments
// Source: https://en.wikipedia.org/wiki/HSL_and_HSV#HSV_to_RGB
__device__ void hsvToRgb(float h, float s, float v, float* r, float* g, float* b) {
    float c = v * s;
    float hp = h / 60.0f;
    float x = c * (1.0f - fabsf(fmodf(hp, 2.0f) - 1.0f));
    float m = v - c;

    float rr = 0.0f;
    float gg = 0.0f;
    float bb = 0.0f;

    if (hp < 1.0f) {
        rr = c;
        gg = x;
    } else if (hp < 2.0f) {
        rr = x;
        gg = c;
    } else if (hp < 3.0f) {
        gg = c;
        bb = x;
    } else if (hp < 4.0f) {
        gg = x;
        bb = c;
    } else if (hp < 5.0f) {
        rr = x;
        bb = c;
    } else {
        rr = c;
        bb = x;
    }

    // Add m to match the value
    // Clamp the final RGB values to [0, 1]
    *r = rr + m;
    *g = gg + m;
    *b = bb + m;
}

__global__ void adjustPixels(unsigned char* imageData, int width, int height, int channels,
                             float brightness, float contrast, float gamma,
                             float hueDegrees, float saturationAdjust, float vibrance,
                             bool grayscale) {

    // Thread index calculation
    int pixelIdx = blockIdx.x * blockDim.x + threadIdx.x;

    // Total number of pixels for bound checking
    int totalPixels = width * height;

    // Each thread processes one pixel, so we check if the pixel index is within bounds
    if (pixelIdx >= totalPixels) {
        return;
    }

    // Calculate the base index for the pixel in the image data array
    int baseIndex = pixelIdx * channels; // 3 channels per pixel for RGB, 4 for RGBA
    if (channels < 3) {
        for (int c = 0; c < channels; ++c) {
            int idx = baseIndex + c;
            float pixelValue = imageData[idx] / 255.0f;

            // Apply contrast (center around 0.5, scale, then shift back)
            pixelValue = (pixelValue - 0.5f) * (1.0f + contrast) + 0.5f;

            // Apply brightness
            pixelValue += brightness;

            // Clamp to [0, 1] before gamma correction
            pixelValue = fmaxf(0.0f, fminf(1.0f, pixelValue));

            // Apply gamma correction if requested
            if (gamma > 0.0f && fabsf(gamma - 1.0f) > 1e-6f) {
                pixelValue = powf(pixelValue, 1.0f / gamma);
            }
            imageData[idx] = (unsigned char)(pixelValue * 255.0f);
        }
        return;
    }

    // Read original RGB values and normalize to [0, 1]
    float r = imageData[baseIndex + 0] / 255.0f;
    float g = imageData[baseIndex + 1] / 255.0f;
    float b = imageData[baseIndex + 2] / 255.0f;
    // Handle alpha channel if present
    float a = (channels == 4) ? imageData[baseIndex + 3] / 255.0f : 1.0f;

    for (int c = 0; c < 3; ++c) {
        int idx = baseIndex + c;
        float pixelValue = imageData[idx] / 255.0f;
        pixelValue = (pixelValue - 0.5f) * (1.0f + contrast) + 0.5f;
        pixelValue += brightness;
        pixelValue = fmaxf(0.0f, fminf(1.0f, pixelValue));
        if (gamma > 0.0f && fabsf(gamma - 1.0f) > 1e-6f) {
            pixelValue = powf(pixelValue, 1.0f / gamma);
        }
        imageData[idx] = (unsigned char)(pixelValue * 255.0f);
    }

    // Convert to HSV for hue, saturation, and vibrance adjustments
    r = imageData[baseIndex + 0] / 255.0f;
    g = imageData[baseIndex + 1] / 255.0f;
    b = imageData[baseIndex + 2] / 255.0f;

    float h, s, v;
    rgbToHsv(r, g, b, &h, &s, &v);

    h = fmodf(h + hueDegrees, 360.0f);
    if (h < 0.0f) {
        h += 360.0f;
    }

    if (saturationAdjust != 0.0f) {
        s *= (1.0f + saturationAdjust);
        s = fmaxf(0.0f, fminf(1.0f, s));
    }

    if (vibrance != 0.0f) {
        if (vibrance > 0.0f) {
            s += (1.0f - s) * vibrance;
        } else {
            s += s * vibrance;
        }
        s = fmaxf(0.0f, fminf(1.0f, s));
    }

    hsvToRgb(h, s, v, &r, &g, &b);

    if (grayscale) {
        float grayValue = 0.299f * r + 0.587f * g + 0.114f * b;
        r = grayValue;
        g = grayValue;
        b = grayValue;
    }

    imageData[baseIndex + 0] = (unsigned char)(fmaxf(0.0f, fminf(1.0f, r)) * 255.0f);
    imageData[baseIndex + 1] = (unsigned char)(fmaxf(0.0f, fminf(1.0f, g)) * 255.0f);
    imageData[baseIndex + 2] = (unsigned char)(fmaxf(0.0f, fminf(1.0f, b)) * 255.0f);
    if (channels == 4) {
        imageData[baseIndex + 3] = (unsigned char)(fmaxf(0.0f, fminf(1.0f, a)) * 255.0f);
    }
}

__global__ void vignetteKernel(unsigned char* imageData, int width, int height, int channels, float intensity){
    int pixelIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = width * height;

    // Boundary check
    if (pixelIdx >= totalPixels) return;

    // Convert 1D thread layout index to 2D coordinates
    int x = pixelIdx % width;
    int y = pixelIdx / width;

    float centerX = width / 2.0f;
    float centerY = height / 2.0f;

    // Euclidean distance calc from center
    float maxDistance = sqrtf(centerX * centerX + centerY * centerY);
    float currentDistance = sqrtf( (x-centerX) * (x - centerX) + (y - centerY) * (y - centerY));

    // Falloff formula
    float factor = 1.0f - (currentDistance / maxDistance) * intensity;
    factor = fmaxf(0.0f, fminf(1.0f, factor));

    int baseIndex = pixelIdx * channels;
    int numChannels = (channels >= 3) ? 3 : channels;
    for (int c = 0; c < numChannels; ++c){
        float val = imageData[baseIndex + c] * factor;
        imageData[baseIndex+c] = (unsigned char)fmax(0.0f, fminf(255.0f, val));
    }
}

// Convolution Kernel: Handles spatial filters (blur and sharpen)
__global__ void convolutionKernel (const unsigned char* inData, unsigned char* outData,
                                    int width, int height, int channels, int mode){
    int pixelIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = width * height;

    // Bound Checking
    if (pixelIdx >= totalPixels) return;

    int x = pixelIdx % width;
    int y = pixelIdx / width;

    // Filter weights configuration matrix
    // NOTE: Larger kernel makes for heavier blur within less passes
    float kernel[3][3];
    if (mode == 0){ // 3x3 box blur
        for (int i = 0; i < 3; ++i){
            for ( int j = 0; j < 3; ++j) kernel[i][j] = 1.0f / 9.0f;
        }
    } else if (mode == 1){ // 3x3 Sharpen kernel
        // NOTE: EDIT s FOR STRONGER SHARPEN
        float s = 2.5f; // Strength, higher value increases sharpening
        kernel[0][0] = 0.0f; kernel[0][1] = -s; kernel[0][2] = 0.0f;
        kernel[1][0] = -s; kernel[1][1] = 1.0f; kernel[1][2] = -s;
        kernel[2][0] = 0.0f; kernel[2][1] = -s; kernel[2][2] = 0.0f;
    }

    int baseIdx = pixelIdx * channels;
    int numChannels = (channels >= 3) ? 3 : channels;

    for (int c = 0; c < numChannels; ++c){
        float sum = 0.0f;
        for (int ky = -1; ky <= 1; ++ky){
            for (int kx = -1; kx <= 1; ++kx){
                int nx = fmaxf(0, fminf(width-1, x+kx));
                int ny = fmaxf(0, fminf(height-1, y+ky));
                int neighborIdx = (ny*width + nx) * channels + c;
                sum += inData[neighborIdx] * kernel[ky + 1][kx + 1];
            }
        }
        outData[baseIdx+c] = (unsigned char)fmaxf(0.0f, fminf(255.0f, sum));
    }
    // Retains transparency data if an alpha channel is present
    if (channels == 4){
        outData[baseIdx + 3] = inData[baseIdx + 3];
    }
}

// Chroma Key remove is pure CUDA Background Removal (no OpenCV Model)
__global__ void chromaKeyKernel(const unsigned char* inputData, unsigned char* outputData, 
                                int width, int height, int inChannels, 
                                float targetHue, float hueTolerance, 
                                float minSaturation, float minBrightness) {
    
    int pixelIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = width * height;
    
    if (pixelIdx >= totalPixels) return;

    int inBaseIndex = pixelIdx * inChannels;
    int outBaseIndex = pixelIdx * 4; // Output must be 4 channels (RGBA) for transparency

    float r = inputData[inBaseIndex + 0] / 255.0f;
    float g = inputData[inBaseIndex + 1] / 255.0f;
    float b = inputData[inBaseIndex + 2] / 255.0f;

    // Convert to HSV using your existing device function
    float h, s, v;
    rgbToHsv(r, g, b, &h, &s, &v);

    // Calculate how close the current pixel's hue is to the target background hue
    float hueDiff = fabsf(h - targetHue);
    
    // Account for the circular nature of Hue (360 degrees)
    if (hueDiff > 180.0f) {
        hueDiff = 360.0f - hueDiff;
    }

    // Check if the pixel matches the background color criteria
    bool isBackground = (hueDiff <= hueTolerance) && 
                        (s >= minSaturation) && 
                        (v >= minBrightness);

    // Copy original RGB values
    outputData[outBaseIndex + 0] = inputData[inBaseIndex + 0];
    outputData[outBaseIndex + 1] = inputData[inBaseIndex + 1];
    outputData[outBaseIndex + 2] = inputData[inBaseIndex + 2];

    // Set Alpha channel: 0 (transparent) for background, 255 (opaque) for subject
    if (isBackground) {
        // RGBA = {0,0,0,0}
        outputData[outBaseIndex + 0] = 0;
        outputData[outBaseIndex + 1] = 0;
        outputData[outBaseIndex + 2] = 0;
        outputData[outBaseIndex + 3] = 0; 
    } else {
        // If the original image already had transparency, preserve it, otherwise 255
        outputData[outBaseIndex + 3] = (inChannels == 4) ? inputData[inBaseIndex + 3] : 255;
    }
}

// Wrapper function to call the CUDA kernel per pixel
void cudaAdjustPixels(unsigned char* d_imageData, int width, int height, int channels,
                       float brightness, float contrast, float gamma,
                       float hue, float saturation, float vibrance,
                       bool grayscale) {
    int totalPixels = width * height;
    int gridSize = (totalPixels + BLOCK_SIZE - 1) / BLOCK_SIZE;

    adjustPixels<<<gridSize, BLOCK_SIZE>>>(d_imageData, width, height, channels,
                                           brightness, contrast, gamma,
                                           hue, saturation, vibrance,
                                           grayscale);

    cudaDeviceSynchronize();

    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
    }
}

// Wrapper for filter executions
void cudaApplyFilters(unsigned char* d_imageData, int width, int height, int channels, const char* filterType){
    int totalPixels = width * height;
    int gridSize = (totalPixels + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int imageSize = totalPixels * channels;
    if (strcmp(filterType, "vignette") == 0) {
        vignetteKernel<<<gridSize, BLOCK_SIZE>>>(d_imageData, width, height, channels, 0.5f); // 0.5f is the default intensity
        cudaDeviceSynchronize();
    } 
    else if (strcmp(filterType, "blur") == 0 || strcmp(filterType, "sharpen") == 0) {
        unsigned char* d_tempData;
        cudaMalloc((void**)&d_tempData, imageSize);
        cudaMemcpy(d_tempData, d_imageData, imageSize, cudaMemcpyDeviceToDevice);

        // NOTE: More iterations = stronger blur/sharpen effect
        int iterations = 10;

        // mode = ? blur : sharpen
        int mode = (strcmp(filterType, "blur") == 0) ? 0 : 1;
        for (int i = 0; i < iterations; ++i){
            // Current image state to temp reading buffer
            cudaMemcpy(d_tempData, d_imageData, imageSize, cudaMemcpyDeviceToDevice);

            // Run Convolution kernel for sharp/blur
            convolutionKernel<<<gridSize, BLOCK_SIZE>>>(d_tempData, d_imageData, width, height, channels, mode);
            cudaDeviceSynchronize();

        }
        cudaFree(d_tempData);
    }

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Filter Error: %s\n", cudaGetErrorString(error));
    }

}

void cudaChromaKey(const unsigned char* d_inputData, unsigned char* d_outputData, 
                   int width, int height, int inChannels, 
                   float targetHue, float hueTolerance, 
                   float minSaturation, float minBrightness) {
    
    int totalPixels = width * height;
    int gridSize = (totalPixels + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // targetHue: 120.0f is standard Green. 240.0f is standard Blue.
    // hueTolerance: 15.0f to 25.0f is usually a good starting point.
    chromaKeyKernel<<<gridSize, BLOCK_SIZE>>>(d_inputData, d_outputData, width, height, inChannels, 
                                              targetHue, hueTolerance, minSaturation, minBrightness);
    
    cudaDeviceSynchronize();
}

__global__ void applyMaskKernel(const unsigned char* inputData, const unsigned char* maskData, 
                                unsigned char* outputData, int width, int height, int inChannels) {
    int pixelIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixelIdx >= width * height) return;

    int inBaseIndex = pixelIdx * inChannels;
    int outBaseIndex = pixelIdx * 4; // Output is RGBA

    // Copy original colors
    outputData[outBaseIndex + 0] = inputData[inBaseIndex + 0];
    outputData[outBaseIndex + 1] = inputData[inBaseIndex + 1];
    outputData[outBaseIndex + 2] = inputData[inBaseIndex + 2];
    
    // Apply the OpenCV mask to the Alpha channel
    outputData[outBaseIndex + 3] = maskData[pixelIdx];
}

// Wrapper function
void cudaApplyMask(const unsigned char* d_inputData, const unsigned char* d_maskData, 
                   unsigned char* d_outputData, int width, int height, int inChannels) {
    int totalPixels = width * height;
    int gridSize = (totalPixels + BLOCK_SIZE - 1) / BLOCK_SIZE;
    applyMaskKernel<<<gridSize, BLOCK_SIZE>>>(d_inputData, d_maskData, d_outputData, width, height, inChannels);
    cudaDeviceSynchronize();
} 