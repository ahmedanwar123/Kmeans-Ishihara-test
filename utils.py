import cv2


def load_image(image_path):
    """
    Load an image from the specified path

    Args:
        image_path: Path to the image file

    Returns:
        image: Loaded image in BGR format
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")
    return image


def display_image(image, window_name="Image"):
    """
    Display an image until a key is pressed

    Args:
        image: Image to display
        window_name: Name for the display window
    """
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_image(image, output_path):
    """
    Save an image to the specified path

    Args:
        image: Image to save
        output_path: Path where the image will be saved

    Returns:
        success: Boolean indicating if the save was successful
    """
    return cv2.imwrite(output_path, image)
