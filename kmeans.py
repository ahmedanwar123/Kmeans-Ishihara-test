import cv2
import numpy as np
from threading import Thread


def assign_labels_thread(start_idx, end_idx, pixels, centroids, labels):
    """
    Multithreaded pixel-to-centroid assignment

    Args:
        start_idx: Starting index for this thread
        end_idx: Ending index for this thread
        pixels: Array of pixel values
        centroids: Current centroid positions
        labels: Array to store the assigned labels
    """
    for i in range(start_idx, end_idx):
        distances = np.linalg.norm(pixels[i] - centroids, axis=1)
        labels[i] = np.argmin(distances)


def kmeans_multithreaded(
    image, K=2, max_iters=10, color_space="Lab", channels=[1], num_threads=8
):
    """
    Perform K-means clustering on an image using multithreading for faster processing

    Args:
        image: Input image (BGR format)
        K: Number of clusters
        max_iters: Maximum number of iterations
        color_space: Color space to use ("BGR", "HSV", "Lab", "YCbCr")
        channels: List of channels to use for clustering
        num_threads: Number of threads to use for parallel processing

    Returns:
        segmented: Segmented image as grayscale
    """
    # Color space conversions
    if color_space == "HSV":
        image_cs = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif color_space == "Lab":
        image_cs = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    elif color_space == "YCbCr":
        image_cs = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    else:
        image_cs = image.copy()

    # Extract selected channels
    img_data = image_cs[:, :, channels]
    pixels = img_data.reshape(-1, len(channels)).astype(np.float32)

    # Initialize centroids randomly
    np.random.seed(42)
    centroids = pixels[np.random.choice(len(pixels), K, replace=False)]

    labels = np.zeros(len(pixels), dtype=np.int32)

    for _ in range(max_iters):
        # Assign labels with multithreading for faster processing
        num_threads = num_threads
        chunk_size = len(pixels) // num_threads
        threads = []
        for t in range(num_threads):
            start = t * chunk_size
            end = len(pixels) if t == num_threads - 1 else (t + 1) * chunk_size
            thread = Thread(
                target=assign_labels_thread,
                args=(start, end, pixels, centroids, labels),
            )
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()

        # Update centroids
        for k in range(K):
            cluster_points = pixels[labels == k]
            if len(cluster_points) > 0:
                centroids[k] = np.mean(cluster_points, axis=0)

    segmented = labels.reshape(image.shape[:2])

    # Map clusters to grayscale intensities
    unique_labels, counts = np.unique(segmented, return_counts=True)
    sorted_indices = np.argsort(-counts)
    grayscale_map = np.zeros(K, dtype=np.uint8)
    for i, idx in enumerate(sorted_indices):
        grayscale_map[idx] = int(255 * (i / (K - 1))) if K > 1 else 255

    final_image = np.zeros_like(segmented, dtype=np.uint8)
    for k in range(K):
        final_image[segmented == k] = grayscale_map[k]

    return final_image
