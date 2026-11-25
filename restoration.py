
import cv2
import numpy as np


def bilateral_filter(image):
    return cv2.bilateralFilter(image, 5, 35, 35)


def median_filter(image):
    return cv2.medianBlur(image, 3)


def clahe_enhancement(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=1.2,
        tileGridSize=(8, 8)
    )

    enhanced_l = clahe.apply(l)

    enhanced = cv2.merge((enhanced_l, a, b))

    return cv2.cvtColor(
        enhanced,
        cv2.COLOR_LAB2BGR
    )


def gamma_correction(image, gamma=1.1):
    lookup_table = np.array([
        ((i / 255.0) ** (1.0 / gamma)) * 255
        for i in range(256)
    ]).astype("uint8")

    return cv2.LUT(image, lookup_table)


def unsharp_masking(image):
    blurred = cv2.GaussianBlur(
        image,
        (5, 5),
        1.0
    )

    return cv2.addWeighted(
        image, 1.2,
        blurred, -0.2,
        0
    )


def laplacian_sharpening(image):
    laplacian = cv2.Laplacian(
        image,
        cv2.CV_64F
    )

    sharpened = image.astype(np.float64) - 0.3 * laplacian

    return np.clip(
        sharpened, 0, 255
    ).astype(np.uint8)


def restore_old_photo_v2(image):
    denoised = bilateral_filter(image)

    enhanced = clahe_enhancement(denoised)

    restored = unsharp_masking(enhanced)

    return restored


def classical_restoration_pipeline(image):

    # Step 1: Remove noise
    denoised = bilateral_filter(image)

    # Step 2: Remove small isolated noise
    denoised = median_filter(denoised)

    # Step 3: Enhance local contrast
    enhanced = clahe_enhancement(denoised)

    # Step 4: Adjust brightness
    enhanced = gamma_correction(enhanced, gamma=1.1)

    # Step 5: Sharpen image
    sharpened = unsharp_masking(enhanced)

    # Step 6: Enhance edges
    restored = laplacian_sharpening(sharpened)

    return restored
