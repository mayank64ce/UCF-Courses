from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
import os
import hashlib

# Define directional deltas for convolution
delx = [1, -1, 0, 0, 1, 1, -1, -1]
dely = [0, 0, 1, -1, 1, -1, 1, -1]

# Some helper methods:


def conv(mat, filter, transpose=False):
    """
    Perform convolution between a matrix and a filter.

    Args:
        mat (numpy.ndarray): Input matrix.
        filter (numpy.ndarray): Convolution filter.
        transpose (bool): If True, transpose the input matrix.

    Returns:
        numpy.ndarray: Convolved matrix.
    """

    k_r = len(filter)
    if transpose:  # set transpose=True to traverse column-wise
        mat = mat.T
    r, c = mat.shape
    out = np.zeros(shape=(r, c))

    for i in range(r):
        for j in range(c):
            # place center of filter on mat[i, j]
            value = 0

            for dy in range(-int(k_r/2), int(k_r/2)+1):
                nx = i
                ny = j + dy
                if in_matrix(r, c, nx, ny):
                    # implicit zero-padding is applied here
                    # if the corresponding index of the matrix does not exist,
                    # we take the matrix value at that
                    # place as zero.
                    value += mat[nx, ny] * filter[int(k_r/2) + dy]

            out[i, j] = value

    if transpose:
        out = out.T

    return out


def in_matrix(r, c, i, j):
    """
    Check if a given index is within matrix bounds.

    Args:
        r (int): Number of rows in the matrix.
        c (int): Number of columns in the matrix.
        i (int): Row index.
        j (int): Column index.

    Returns:
        bool: True if the index is within bounds, False otherwise.
    """

    return (0 <= i and i < r) and (0 <= j and j < c)


def get_directions(angle):
    """
    Determine directional deltas based on an angle.

    Args:
        angle (float): Angle in degrees.

    Returns:
        tuple: Directional deltas (delx, dely).
    """

    if (-22.5 < angle <= 22.5) or (-157.5 < angle <= 157.5):                     # Horizontal direction
        return [0, 0], [-1, 1]
    elif (-112.5 < angle <= -67.5) or (67.5 < angle <= 112.5):                   # Veritcal direction
        return [1, -1], [0, 0]
    elif (-67.5 < angle <= -22.5) or (112.5 < angle <= 157.5):                   # -45 direction
        return [-1, 1], [1, -1]
    else:                                                                        # +45 direction
        return [1, -1], [1, -1]

# Step 1: Gaussian Blur


def get_gaussian(sigma=1, mu=0, dim=25, normalize=True):
    """
    Generate a Gaussian filter for blurring.

    Args:
        sigma (float): Standard deviation of the Gaussian.
        mu (float): Mean of the Gaussian.
        dim (int): Dimension of the filter (must be odd).
        normalize (bool): If True, normalize the filter.

    Returns:
        numpy.ndarray: Gaussian filter.
    """

    assert sigma > 0
    assert dim % 2 != 0

    r = np.linspace(-int(dim/2), int(dim/2), dim)
    gaussian_filter = np.exp(-((r-mu)**2)/(2*sigma**2))

    if normalize:
        gaussian_filter /= sigma*math.sqrt(2*math.pi)

    return gaussian_filter


# Step 2: Derivative of Gaussian blur

def get_gaussian_derivative(sigma=1, mu=0, dim=25, normalize=True):
    """
    Generate a Gaussian derivative filter.

    Args:
        sigma (float): Standard deviation of the Gaussian.
        mu (float): Mean of the Gaussian.
        dim (int): Dimension of the filter (must be odd).
        normalize (bool): If True, normalize the filter.

    Returns:
        numpy.ndarray: Gaussian derivative filter.
    """

    r = np.linspace(-int(dim/2), int(dim/2), dim)
    result = -(r-mu) * get_gaussian(sigma, mu, dim, normalize)
    if normalize:
        result /= (sigma**2)
    return result


# Step 3: Get Magnitude and orientation of Gradients:

def get_magnitude_matrix(gx, gy):
    """
    Calculate the magnitude of gradients.

    Args:
        gx (numpy.ndarray): Gradient in the x-direction.
        gy (numpy.ndarray): Gradient in the y-direction.

    Returns:
        numpy.ndarray: Magnitude of gradients.
    """

    assert gx.shape == gy.shape
    out = np.zeros(shape=gx.shape)

    r, c = gx.shape

    for i in range(r):
        for j in range(c):
            out[i, j] = math.sqrt(gx[i, j]**2 + gy[i, j]**2)

    # print(out.min(), out.max())

    out = out / out.max() * 255

    return out


def get_angle_matrix(gx, gy):
    """
    Calculate the orientation of gradients.

    Args:
        gx (numpy.ndarray): Gradient in the x-direction.
        gy (numpy.ndarray): Gradient in the y-direction.

    Returns:
        numpy.ndarray: Orientation of gradients.
    """

    assert gx.shape == gy.shape
    out = np.zeros(shape=gx.shape)

    r, c = gx.shape

    for i in range(r):
        for j in range(c):
            out[i, j] = math.atan2(gy[i, j], gx[i, j])*(180/math.pi)

    return out


# Step 4: Perform non-maximum suppression

def get_nonmax_suppression(M, theta):
    """
    Perform non-maximum suppression.

    Args:
        M (numpy.ndarray): Magnitude of gradients.
        theta (numpy.ndarray): Orientation of gradients.

    Returns:
        numpy.ndarray: Suppressed image.
    """

    r, c = M.shape
    out = M.copy()

    for i in range(r):
        for j in range(c):
            delx, dely = get_directions(theta[i, j])

            for k in range(len(delx)):
                dx = delx[k]
                dy = dely[k]

                nx, ny = i + dx, j + dy

                if in_matrix(r, c, nx, ny):
                    # if any neighbour in direction of orientation has bigger magnitude, su
                    if M[nx, ny] > M[i, j]:
                        # suppress this pixel
                        out[i, j] = 0

    return out


# Step 5: perform double thresholding and hysterisis

def double_threshold(image, low_threshold, high_threshold):
    """
    Perform double thresholding on an image.

    Args:
        image (numpy.ndarray): Input image.
        low_threshold (float): Low threshold value.
        high_threshold (float): High threshold value.

    Returns:
        numpy.ndarray: Thresholded image.
    """

    r, c = image.shape

    output = np.zeros((r, c))

    weak = 105
    strong = 255

    low_threshold = low_threshold * 255
    high_threshold = high_threshold * 255

    for i in range(r):
        for j in range(c):
            if image[i, j] > high_threshold:
                output[i, j] = strong
            elif image[i, j] < low_threshold:
                output[i, j] = 0
            else:
                output[i, j] = weak

    return output


def get_hysterisis(img, labels):
    """
    Apply hysterisis thresholding to an image.

    Args:
        img (numpy.ndarray): Input image.
        labels (numpy.ndarray): Image labels.

    Returns:
        numpy.ndarray: Hysterisis thresholded image.
    """

    out = img.copy()

    weak = 105
    strong = 255

    r, c = img.shape

    labels_of_strong = set()

    for i in range(r):
        for j in range(c):
            if img[i, j] == strong:
                # gathering the labels for strong pixels
                labels_of_strong.add(labels[i, j])

    for i in range(r):
        for j in range(c):
            if img[i, j] == weak:

                if labels[i, j] in labels_of_strong:
                    # if a weak pixel is in the same connected component as
                    # a strong pixel, set it to high value
                    out[i, j] = strong
                else:
                    out[i, j] = 0

    return out


def CannyEdgeDetector(image, sigma, low_thresh, high_thresh, dim=3, mu=0, plot=False, return_all=False):
    """
    Apply the Canny edge detection algorithm to an image.

    Args:
        image (PIL.Image.Image): Input image.
        sigma (float): Standard deviation for Gaussian blur.
        low_thresh (float): Low threshold for edge detection.
        high_thresh (float): High threshold for edge detection.
        dim (int): Dimension of filters.
        mu (float): Mean for filters.
        plot (bool): If True, display intermediate results.
        return_all (bool): If True, return all intermediate images.

    Returns:
        PIL.Image.Image: Edge-detected image.
    """

    I = np.array(image)

    normalize = True

    gaussian_filter = get_gaussian(
        sigma=sigma, normalize=normalize, dim=dim, mu=mu)
    gaussian_derivative_filter = get_gaussian_derivative(
        sigma=sigma, normalize=normalize, dim=dim, mu=mu)

    I_x = conv(I, gaussian_filter)
    I_y = conv(I, gaussian_filter, transpose=True)

    I_prime_x = conv(I_x, gaussian_derivative_filter)
    I_prime_y = conv(I_y, gaussian_derivative_filter, transpose=True)

    M = get_magnitude_matrix(I_prime_x, I_prime_y)

    theta = get_angle_matrix(I_prime_x, I_prime_y)

    I_nms = get_nonmax_suppression(M, theta)

    I_thresh = double_threshold(
        I_nms, low_threshold=low_thresh, high_threshold=high_thresh).astype(np.uint8)

    _, labels = cv2.connectedComponents(
        I_thresh, connectivity=8)  # get connected components

    I_hyst = get_hysterisis(I_thresh, labels)

    # breakpoint()

    if plot:
        plt.subplot(3, 3, 1)
        plt.axis('off')
        plt.imshow(I_x, cmap="gray")
        plt.title("Gaussian blurred in x")

        plt.subplot(3, 3, 2)
        plt.axis('off')
        plt.imshow(I_y, cmap="gray")
        plt.title("Gaussian blurred in y")

        plt.subplot(3, 3, 3)
        plt.axis('off')
        plt.imshow(I_prime_x, cmap="gray")
        plt.title("Gaussian blurred derivative in x")

        plt.subplot(3, 3, 4)
        plt.axis('off')
        plt.imshow(I_prime_y, cmap="gray")
        plt.title("Gaussian blurred derivative in y")

        plt.subplot(3, 3, 5)
        plt.axis('off')
        plt.imshow(M, cmap="gray")
        plt.title("Magnitude of gradient")

        plt.subplot(3, 3, 6)
        plt.axis('off')
        plt.imshow(I_nms, cmap="gray")
        plt.title("Nonmax Suppression")

        plt.subplot(3, 3, 8)
        plt.axis('off')
        plt.imshow(I_hyst, cmap="gray")
        plt.title("Hysterisis Thresholding")

        plt.show()

    if return_all:
        return I_x, I_y, I_prime_x, I_prime_y, M, I_nms, I_thresh, I_hyst

    return I_hyst


def get_results(images, sigmas):

    output_dir_prime = "output"

    for image in images:
        image = os.path.join("input", image)
        img = Image.open(image)
        img_name, extension = os.path.basename(image).split('.')

        output_dir_image = os.path.join(output_dir_prime, img_name)

        for sigma in sigmas:
            I_x, I_y, I_prime_x, I_prime_y, M, I_nms, I_hyst = CannyEdgeDetector(
                img, sigma, 0.1, 0.2, dim=11, return_all=True, plot=True)

            # print(I_hyst.max(), I_hyst.min())

            output_dir = os.path.join(output_dir_image, f"sigma={sigma}")

            os.makedirs(output_dir, exist_ok=True)

            I_x = Image.fromarray(I_x.astype(np.uint8))
            I_y = Image.fromarray(I_y.astype(np.uint8))
            I_prime_x = Image.fromarray(I_prime_x.astype(np.uint8))
            I_prime_y = Image.fromarray(I_prime_y.astype(np.uint8))
            M = Image.fromarray(M.astype(np.uint8))
            I_nms = Image.fromarray(I_nms.astype(np.uint8))
            I_hyst = Image.fromarray(I_hyst.astype(np.uint8))

            breakpoint()

            # save outputs

            I_x.save(os.path.join(
                output_dir, f'blurred_x_sigma={sigma}.{extension}'))
            I_y.save(os.path.join(
                output_dir, f'blurred_y_sigma={sigma}.{extension}'))
            I_prime_x.save(os.path.join(
                output_dir, f'derivative_x_sigma={sigma}.{extension}'))
            I_prime_y.save(os.path.join(
                output_dir, f'derivative_y_sigma={sigma}.{extension}'))
            M.save(os.path.join(
                output_dir, f'magnitude_sigma={sigma}.{extension}'))
            I_nms.save(os.path.join(
                output_dir, f'nms_sigma={sigma}.{extension}'))
            I_hyst.save(os.path.join(
                output_dir, f'hyst_sigma={sigma}.{extension}'))

            # plt.imshow(I_prime_x, cmap="gray")
            # plt.axis('off')
            # # plt.savefig(f'{img_name}_edge.{extension}',
            # #             bbox_inches='tight', pad_inches=0)
            # plt.show()


def visualize_images(images, sigmas):

    for image_path in images:
        image_path = os.path.join("input", image_path)
        for sigma in sigmas:
            fig = visualize_one(image_path=image_path, sigma=sigma,
                                low_thresh=0.1, high_thresh=0.2)
            # print("hello")
            plt.show()


def visualize_one(image_path, sigma, low_thresh, high_thresh):

    img = Image.open(image_path)

    img_name = os.path.basename(image_path)

    image_list = list(CannyEdgeDetector(
        img, sigma, low_thresh=low_thresh, high_thresh=high_thresh, dim=5, return_all=True))

    image_list = [img] + image_list

    # print(image_list[0].shape)

    fig, axes = plt.subplots(3, 3, figsize=(8, 8))

    # print("hello")

    # Names for the subplots
    subplot_names = [
        "(a) Original Image",
        "(b) Gaussian blurred in x",
        "(c) Gaussian blurred in y",
        "(d) Gaussian blurred derivative in x",
        "(e) Gaussian blurred derivative in y",
        "(f) Magnitude of gradient",
        "(g) Nonmax Suppression",
        "(h) Double Thresholding",
        "(i) Hysterisis",
    ]

    # Create some example image data
    for i, ax in enumerate(axes.flatten()):
        # Display image in the current subplot
        if i > len(image_list)-1:
            print(len(image_list))
            break
        ax.imshow(image_list[i], cmap='gray')
        # print(i)
        ax.set_title(subplot_names[i])
        ax.axis("off")

    # Set the overall figure title
    fig.suptitle(f'img = {img_name} sigma={sigma}')

    return fig


if __name__ == "__main__":

    image_path = "input/gray1.jpg"  # replace this filepath to desired image path

    img = Image.open(image_path)

    edges = CannyEdgeDetector(img, sigma=2, dim=11,
                              low_thresh=0.1, high_thresh=0.2)

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap="gray")
    plt.axis("off")
    plt.title("Canny edge detection output")
    plt.show()
