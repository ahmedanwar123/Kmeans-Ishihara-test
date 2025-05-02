from kmeans import kmeans_multithreaded
from utils import load_image, display_image, save_image
from config import (
    INPUT_IMAGE_PATH,
    OUTPUT_IMAGE_PATH,
    K_CLUSTERS,
    MAX_ITERATIONS,
    COLOR_SPACE,
    CHANNELS,
    NUM_THREADS,
)


def main():
    # Load the image
    image = load_image(INPUT_IMAGE_PATH)

    # Perform k-means segmentation
    segmented = kmeans_multithreaded(
        image,
        K=K_CLUSTERS,
        max_iters=MAX_ITERATIONS,
        color_space=COLOR_SPACE,
        channels=CHANNELS,
        num_threads=NUM_THREADS,
    )

    # Display the result
    display_image(segmented, "Segmented Output")

    # Save the result
    save_image(segmented, OUTPUT_IMAGE_PATH)
    print(f"Segmented image saved to {OUTPUT_IMAGE_PATH}")


if __name__ == "__main__":
    main()
