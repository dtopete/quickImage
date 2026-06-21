#let accent = rgb("0f766e")
#let accent-soft = rgb("99f6e4")

#set page(
	margin: (top: 0.85in, bottom: 0.9in, left: 0.9in, right: 0.9in),
)

#set text(
	font: "Libertinus Serif",
	size: 12pt,
	fill: rgb("1f2937"),
)

#set par(
	justify: true,
	leading: 0.55em,
)

#show heading.where(level: 1): it => block(below: 0.3em)[
	#text(size: 18pt, weight: "bold", fill: accent)[#it.body]
	#line(length: 100%, stroke: 1.1pt + accent-soft)
]

#show heading.where(level: 2): it => block(below: 0.4em)[
	#text(size: 15pt, weight: "semibold", fill: accent)[#it.body]
]

#show heading.where(level: 3): it => block(below: 0.4em)[
	#text(size: 13pt, weight: "semibold", fill: accent)[#it.body]
]

#show raw: set text(font: "Iosevka", size: 12pt)

#align(center)[
	#text(size: 24pt, weight: "bold", fill: accent)[Project Proposal: quickImage]

	#v(0.25em)
	#text(size: 12pt)[Danny Topete - dtope004]

	#text(size: 12pt, fill: rgb("4b5563"))[EE/CS 147]

	#v(0.75em)
	#line(length: 58%, stroke: 1pt + accent-soft)
]

#v(0.9em)

= Required Libraries and Frameworks
- *OpenCV*: for image processing and applying filters. Aid with background removal and HEIC support. Allowing for computer vision techniques for advanced filters and background removal.
- *CUDA Toolkit*: GPU acceleration and running the C++ kernels.
- *std_image* / *stb_image_write*: for loading JPGs and PNGs into raw byte arrays to copy to the GPU (device memory) and writing the output back to disk.
- *Argparse* (C++): for parsing command line arguments such as flags.

= Potential Risks or Challenges
- *Background Removal*: Implementing accurate background removal can be challenging, especially for complex images. I may focus on portraits for this feature. I believe I can use convolution for this technique, but it may require experimentation and fine-tuning to achieve good results.
- *HEIC Support*: Most phones now save photos in HEIC format, so I want to support this, but JPG and PNG will be my priority, then I will add HEIC support if I have time.
- *Warp Divergence*: Using filters such as blur and edge detection will lead to warp divergence due to necessary conditional statements. Requiring optimization and experimentation to achieve good performance.
- *Boundaries and Edge Cases*: Handling blur where the pixel doesn't have a nearby pixel to sample from. Handling the padding in parallel threads will be tricky and may cause artifacts.

#pagebreak()

= Plan/Outline for Project
== Gist
quickImage is a CLI-based tool with GPU-accelerated image filters.

== Motivation
I find myself opening Photoshop or Lightroom for very basic edits to my photos. I want a tool that can do those basic edits without the overhead of opening a large application and the acceleration of GPUs. quickImage will be a command line tool that can apply filters to images using the GPU for acceleration.

Inspired by the ease of use of ffmpeg, quickImage will have a simple syntax for applying filters to images. For example, to increase the brightness of an image, you could run:
```typst
quickImage input.jpg  --brightness 0.2 --contrast 0.1
```

== Features
=== Basic Features
- GPU acceleration for faster processing. Maybe have a live preview of the filter being applied to the image if you are using the CLI menu.
- Basic filters: brightness, contrast, saturation, hue, etc.
- Batch processing: apply filters to multiple images at once.
- Support for common image formats: JPEG, PNG, HEIC, etc.

=== Advanced Features
- Background detection and removal.
	- Remove background from portraits using edge detection and segmentation techniques.
	- Return a PNG with a transparent background.
	- Add a glow effect to the edges of the subject for a more artistic look.
- Filters.
  - Apply LUTs (lookup tables) for color grading and creative effects.
	- Advanced filters: blur, sharpen, gaussian blur, edge detection, etc.
	- Add Fuji Film's film simulation filters to give photos a classic look including film grain, color grading, and tone curve adjustments.
	- Add a filter that can mimic the look of vintage cameras, such as the Holga or Diana, which are known for their unique lens distortions and light leaks.
- Video support: apply filters to videos as well as images.
	- This would be a stretch goal, but it would be a great feature to have. It would allow users to apply filters to their videos in the same way they do with images.