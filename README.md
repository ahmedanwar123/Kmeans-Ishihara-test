
# K-Means Image Segmentation

This project implements a multithreaded K-means clustering algorithm for image segmentation, particularly useful for analyzing Ishihara color blindness test plates.

## Project Structure

- `main.py`: The entry point of the application
- `kmeans.py`: Contains the K-means clustering algorithm implementation
- `utils.py`: Utility functions for image handling (loading, displaying, saving)
- `config.py`: Configuration settings for the segmentation process

## Features

- Multithreaded implementation for faster processing
- Support for different color spaces (BGR, HSV, Lab, YCbCr)
- Flexible channel selection for clustering
- Configurable number of clusters and iterations

## Usage

1. Update the configuration in `config.py` with your image path and desired parameters
2. Run the application:

```bash
python main.py
```

## Configuration Options

- `INPUT_IMAGE_PATH`: Path to the input image
- `OUTPUT_IMAGE_PATH`: Path where the segmented image will be saved
- `K_CLUSTERS`: Number of clusters for segmentation
- `MAX_ITERATIONS`: Maximum number of iterations for the K-means algorithm
- `COLOR_SPACE`: Color space to use ("BGR", "HSV", "Lab", "YCbCr")
- `CHANNELS`: Channels to use for clustering (e.g., [1] for the second channel)
- `NUM_THREADS`: Number of threads to use for parallel processing

## Algorithm

The implemented K-means algorithm:

1. Converts the image to the specified color space
2. Extracts the specified channels for clustering
3. Randomly initializes K centroids
4. Assigns each pixel to the nearest centroid using multithreaded processing
5. Updates centroids based on assigned pixels
6. Repeats steps 4-5 for a specified number of iterations
7. Maps clusters to grayscale values for visualization

## Requirements

- OpenCV (cv2)
- NumPy
- Python 3.6+
