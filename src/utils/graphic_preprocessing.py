"""
Image preprocessing techniques for skin cancer images.
Includes hair removal, contrast enhancement, and segmentation.
"""

import cv2
import numpy as np


def remove_hair(image, ksize=17, threshold=10):
    """
    Remove hair artifacts from skin lesion images.

    Args:
        image (numpy.array): BGR image.
        ksize (int): Kernel size for morphological operations.
        threshold (int): Threshold value for hair detection.

    Returns:
        numpy.array: Image with hair removed.
    """
    # Convert to grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create kernel for morphological filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))

    # Apply blackhat filter to identify dark regions (potentially hair)
    blackhat = cv2.morphologyEx(grayscale, cv2.MORPH_BLACKHAT, kernel)

    # Threshold to create binary mask of hair
    _, mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)

    # Inpaint using the mask to replace hair with nearby skin pixels
    result = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    return result


def enhance_contrast(image):
    """
    Enhance contrast in skin lesion images using CLAHE.

    Args:
        image (numpy.array): BGR image.

    Returns:
        numpy.array: Contrast-enhanced image.
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    # Merge channels and convert back to BGR
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    return enhanced


def generate_adaptive_circular_contour(image, num_points=100, margin=1.1):
    """
    Generate a circular contour adapted to the lesion based on Otsu + bounding box.

    Args:
        image (numpy.array): BGR image.
        num_points (int): Number of points in the contour.
        margin (float): Margin factor for the contour.

    Returns:
        numpy.array: Circular contour as (num_points, 2) array.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.bitwise_not(thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        # Fallback to central circle
        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        radius = min(h, w) * 0.2

        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)

        contour = np.stack([xs, ys], axis=1).astype(np.float32)
        return contour

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    center = (x + w / 2, y + h / 2)
    a = (w / 2) * margin  # semi-axis x
    b = (h / 2) * margin  # semi-axis y

    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    xs = center[0] + a * np.cos(angles)
    ys = center[1] + b * np.sin(angles)

    contour = np.stack([xs, ys], axis=1).astype(np.float32)
    return contour


def compute_gvf(image, mu=0.2, iterations=80, delta_t=1):
    """
    Compute the gradient vector flow (GVF) for an image.

    Args:
        image (numpy.array): Input image (RGB or grayscale).
        mu (float): Regularization parameter controlling smoothing vs. edge fidelity.
        iterations (int): Number of iterations for field evolution.
        delta_t (float): Time step for each iteration.

    Returns:
        tuple: (u, v) components of the GVF field.
    """
    # Convert to grayscale if necessary
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    gray = gray.astype(np.float32) / 255.0

    # Apply CLAHE for better edge detection
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply((gray * 255).astype(np.uint8)).astype(np.float32) / 255.0

    # Calculate gradients using Sobel
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    # Non-linear edge function
    f = cv2.GaussianBlur(grad_mag, (5, 5), 0)

    # Edge map gradients
    f_grad_x = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=5)
    f_grad_y = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=5)

    # Initialize GVF with edge map gradients
    u = f_grad_x.copy()
    v = f_grad_y.copy()

    # Calculate and normalize penalty term
    squared_mag = f_grad_x ** 2 + f_grad_y ** 2
    squared_mag = squared_mag / (np.max(squared_mag) + 1e-8)

    # Iterative GVF computation
    for _ in range(iterations):
        u_lap = cv2.Laplacian(u, cv2.CV_32F)
        v_lap = cv2.Laplacian(v, cv2.CV_32F)

        u += delta_t * (mu * u_lap - (u - f_grad_x) * squared_mag)
        v += delta_t * (mu * v_lap - (v - f_grad_y) * squared_mag)

    # Handle invalid values
    u = np.nan_to_num(u, nan=0.0, posinf=1.0, neginf=-1.0)
    v = np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=-1.0)

    return u, v


def interp_gvf_vectorized(contour, u, v):
    """
    Perform vectorized bilinear interpolation of the GVF field for a set of points.

    Args:
        contour (numpy.array): Array of points with shape (N, 2).
        u (numpy.array): x-component of the GVF field.
        v (numpy.array): y-component of the GVF field.

    Returns:
        numpy.array: Interpolated GVF vectors at contour points.
    """
    h, w = u.shape

    # Clip coordinates within bounds for interpolation
    x = np.clip(contour[:, 0], 0, w - 2)
    y = np.clip(contour[:, 1], 0, h - 2)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)

    dx = x - x0
    dy = y - y0

    # Vectorized bilinear interpolation
    u_interp = (u[y0, x0] * (1 - dx) * (1 - dy) +
                u[y0, x1] * dx * (1 - dy) +
                u[y1, x0] * (1 - dx) * dy +
                u[y1, x1] * dx * dy)

    v_interp = (v[y0, x0] * (1 - dx) * (1 - dy) +
                v[y0, x1] * dx * (1 - dy) +
                v[y1, x0] * (1 - dx) * dy +
                v[y1, x1] * dx * dy)

    return np.column_stack([u_interp, v_interp])


def gvf_based_segmentation(image, init_contour=None, mu=0.2, iterations=80, delta_t=1,
                           alpha=0.1, beta=0.1, gamma=1, kappa=0.5, iterations_snake=100):
    """
    Perform GVF-based active contour (snake) segmentation.

    Args:
        image (numpy.array): Input image (BGR or grayscale).
        init_contour (numpy.array, optional): Initial contour points. If None, creates one.
        mu (float): Regularization parameter for GVF.
        iterations (int): Number of iterations for GVF computation.
        delta_t (float): Time step for GVF computation.
        alpha (float): Tension weight (contour energy).
        beta (float): Rigidity weight (bending energy).
        gamma (float): Step size for internal force.
        kappa (float): Weight of external force from GVF.
        iterations_snake (int): Number of iterations for snake evolution.

    Returns:
        numpy.array: Final contour after evolution.
    """
    # Generate initial contour if not provided
    if init_contour is None:
        init_contour = generate_adaptive_circular_contour(image)

    # Compute GVF
    u, v = compute_gvf(image, mu=mu, iterations=iterations, delta_t=delta_t)

    # Initialize contour
    contour = init_contour.copy()
    h, w = u.shape

    # Snake evolution
    for _ in range(iterations_snake):
        # Get neighbors using np.roll (closed contour)
        prev = np.roll(contour, 1, axis=0)
        next = np.roll(contour, -1, axis=0)

        # Compute internal forces
        tension = alpha * (next - 2 * contour + prev)
        rigidity = beta * ((next - contour) - (contour - prev))
        internal_force = tension + rigidity

        # Compute external force from GVF
        external_force = interp_gvf_vectorized(contour, u, v)

        # Update contour
        contour += gamma * internal_force + kappa * external_force

        # Keep points within image bounds
        contour[:, 0] = np.clip(contour[:, 0], 0, w - 1)
        contour[:, 1] = np.clip(contour[:, 1], 0, h - 1)

    return contour


def apply_segmentation(image):
    """
    Apply GVF-based segmentation to an image and visualize the contour.

    Args:
        image (numpy.array): BGR image.

    Returns:
        numpy.array: Image with segmentation contour drawn.
    """
    # Generate initial contour
    init_contour = generate_adaptive_circular_contour(image, num_points=100)

    # Perform segmentation
    final_contour = gvf_based_segmentation(
        image,
        init_contour,
        mu=0.05,
        iterations=200,  # Reduced for faster processing
        delta_t=0.1,
        alpha=0.05,
        beta=0.01,
        gamma=1,
        kappa=10,
        iterations_snake=250  # Reduced for faster processing
    )

    # Convert to integer points for drawing
    final_contour_int = final_contour.reshape((-1, 1, 2)).astype(np.int32)

    # Draw contour on image
    segmented_image = image.copy()
    cv2.drawContours(segmented_image, [final_contour_int], contourIdx=-1,
                     color=(0, 0, 255), thickness=2)

    return segmented_image


def create_lesion_mask(image):
    """
    Create a binary mask of the skin lesion using Otsu thresholding.

    Args:
        image (numpy.array): BGR image.

    Returns:
        numpy.array: Binary mask where lesion pixels are 255.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Otsu's thresholding
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find the largest contour (assumed to be the lesion)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.zeros_like(gray)

    # Create a new mask with only the largest contour
    lesion_mask = np.zeros_like(gray)
    cv2.drawContours(lesion_mask, [max(contours, key=cv2.contourArea)], 0, 255, -1)

    return lesion_mask


def apply_graphic_preprocessing(image, use_hair_removal=True, use_contrast_enhancement=True,
                                use_segmentation=False, visualize=False):
    """
    Apply multiple preprocessing techniques to enhance skin lesion images.

    Args:
        image (numpy.array): BGR image.
        use_hair_removal (bool): Whether to apply hair removal.
        use_contrast_enhancement (bool): Whether to enhance contrast.
        use_segmentation (bool): Whether to apply and visualize segmentation.
        visualize (bool): Whether to visualize processing steps.

    Returns:
        numpy.array: Processed image.
    """
    original = image.copy()
    processed = image.copy()

    # Apply preprocessing steps in sequence
    if use_hair_removal:
        processed = remove_hair(processed)

    if use_contrast_enhancement:
        processed = enhance_contrast(processed)

    if use_segmentation:
        processed = apply_segmentation(processed)

    if visualize:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title('Original')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
        plt.title('Processed')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    return processed