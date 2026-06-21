#!/bin/bash

# Setup script to download STB image libraries
echo "Downloading stb_image.h..."
curl -sS https://raw.githubusercontent.com/nothings/stb/master/stb_image.h -o stb_image.h

echo "Downloading stb_image_write.h..."
curl -sS https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h -o stb_image_write.h

if [ -f stb_image.h ] && [ -f stb_image_write.h ]; then
    echo "✓ Successfully downloaded STB libraries!"
    echo ""
    echo "You can now build the project with: make"
else
    echo "✗ Failed to download STB libraries."
    echo "Please ensure you have curl installed and internet access."
    exit 1
fi
