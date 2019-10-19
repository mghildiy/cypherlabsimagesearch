import numpy as np
import cv2
from skimage.exposure import rescale_intensity
import argparse

def convolve(image, kernel):
    (iHeight, iWidth) = image.shape[:2]
    (kHeight, kWidth) = kernel.shape[:2]

    # apply pad to original image to compensate for size reduction caused by convolution
    pad = (kWidth - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

    # initialize output
    output = np.zeros(iHeight, iWidth, "float")

    # loop over the input image(first along width, then along height), "sliding" the kernel across
    for h in np.arange(pad, iHeight + pad):
        for w in np.arange(pad, iWidth + pad):
            # extract the region of image on which we apply convolution. (h,w) is centre of that region.
            roi = image[h-pad:h+pad+1, w-pad:w+pad+1]
            # multiply kernel and ROI, and take sum
            k = (roi * kernel).sum()
            output[h-pad, w-pad] = k

    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")

    return output

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

# construct average blurring kernels used to smooth an image
smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))

# construct a sharpening kernel
sharpen = np.array((
                    [0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]), dtype="int")

# construct the Laplacian kernel used to detect edge-like regions of an image
laplacian = np.array((
                    [0, 1, 0],
                    [1, -4, 1],
                    [0, 1, 0]), dtype="int")

# construct the Sobel x-axis kernel
sobelX = np.array((
                    [-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]), dtype="int")

# construct the Sobel y-axis kernel
sobelY = np.array((
                    [-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]), dtype="int")

# construct an emboss kernel
emboss = np.array((
                    [-2, -1, 0],
                    [-1, 1, 1],
                    [0, 1, 2]), dtype="int")

# construct the kernel bank, a list of kernels we’re going to apply
# using both our custom 'convolve' function and OpenCV’s ‘filter2D‘ function
kernelBank = (
        ("small_blur", smallBlur),
        ("large_blur", largeBlur),
        ("sharpen", sharpen),
        ("laplacian", laplacian),
        ("sobel_x", sobelX),
        ("sobel_y", sobelY),
        ("emboss", emboss))

# read the image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# loop over the kernels
for (kernelName, K) in kernelBank:
    # apply the kernel to the grayscale image using both our custom
    # ‘convolve‘ function and OpenCV’s ‘filter2D‘ function
    print("[INFO] applying {} kernel".format(kernelName))
    convolveOutput = convolve(gray, K)
    opencvOutput = cv2.filter2D(gray, -1, K)

    # show the output images
    cv2.imshow("Original", gray)
    cv2.imshow("{} - convole".format(kernelName), convolveOutput)
    cv2.imshow("{} - opencv".format(kernelName), opencvOutput)
    cv2.waitKey(0)
    cv2.destroyAllWindows()