import os
import argparse
from scipy.ndimage import binary_fill_holes
from scipy.spatial import ConvexHull
from skimage import img_as_ubyte
from skimage.color import rgb2gray, rgb2hsv
from skimage.filters import sobel
from skimage.morphology import binary_dilation, binary_erosion
from skimage.morphology import disk
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.measure import find_contours
from skimage.filters import threshold_otsu
from skimage.measure import find_contours, approximate_polygon
from skimage import img_as_ubyte
from skimage.color import rgb2gray, rgb2hsv
from skimage import io
from matplotlib import pyplot as plt
import numpy as np


def to_1channel(img, mode='blue'):
    if mode == 'red':
        return img[:, :, 0]
    elif mode == 'green':
        return img[:, :, 1]
    elif mode == 'blue':
        return img[:, :, 2]
    elif mode == 'gray':
        return img_as_ubyte(rgb2gray(img))
    elif mode == 'hue':
        return img_as_ubyte(rgb2hsv(img)[:, :, 0])
    elif mode == 'saturation':
        return img_as_ubyte(rgb2hsv(img)[:, :, 1])
    elif mode == 'value':
        return img_as_ubyte(rgb2hsv(img)[:, :, 2])


def mask_to_remove_noise(image, hole_size=120):
    img_hue = to_1channel(image,'hue')
    mask = (img_hue > 125) & (img_hue < 170)

    return remove_small_holes(mask, hole_size)


def find_binary_borders(image, threshold=6):
    borders = img_as_ubyte(sobel(to_1channel(image, 'gray')))

    return borders > threshold


def mask_cards(image, selem=disk(4)):
    mask = mask_to_remove_noise(image)
    cards = to_1channel(image, 'saturation') * mask
    cards = binary_fill_holes(cards > threshold_otsu(cards))

    return binary_erosion(cards, selem)


def clean_figures(figures, size=1050):
    figures = binary_erosion(figures)
    figures_cleaned = binary_erosion(remove_small_objects(figures, size))
    figures_cleaned = remove_small_objects(figures_cleaned)

    return figures_cleaned


def get_figures(image):
    borders = find_binary_borders(image)
    cards_mask = mask_cards(image)
    figures = binary_erosion(binary_fill_holes(borders * cards_mask))

    return clean_figures(figures)


def find_angles(vectors):
    vectors = np.vstack((vectors[-1], vectors))
    cos = (vectors[1:] * -vectors[:-1]).sum(axis=1)
    angles = np.arccos(cos)

    return angles


def regularized_count(coords, min_dist=5, max_angle=np.pi - 0.15):
    vectors = coords[1:] - coords[:-1]
    dists = np.linalg.norm(vectors, axis=1)
    vectors = vectors / dists[:, None]
    angles = find_angles(vectors)
    mask = (angles < max_angle) * (dists > min_dist)

    return mask.sum()


def find_angles_sum(coords):
    vectors = coords[1:] - coords[:-1]
    dists = np.linalg.norm(vectors, axis=1)
    vectors = vectors / dists[:, None]

    return find_angles(vectors).sum()


def create_labels(figures):
    label_lst = []
    contour_lst = []
    fig_count = 0

    for i, contour in enumerate(find_contours(figures, 0)):
        coords = approximate_polygon(contour, tolerance=6.5)
        if coords.shape[0] <= 2:
            continue
        fig_count +=1
        coords_tight = approximate_polygon(contour, tolerance=2.7)
        angle_coef= find_angles_sum(coords_tight) / coords_tight.shape[0]
        if angle_coef < 2.1:
            n_points = regularized_count(coords, 8, np.pi - 0.15)
            text_label = 'P' + str(n_points)
            hull = ConvexHull(coords)

            if hull.vertices.shape[0] == n_points:
                text_label += 'C'

            point = [np.max(coords[:, 0]), np.min(coords[:, 1])]
            label_lst.append((point, text_label))
        contour_lst.append(contour)

    return label_lst, contour_lst, fig_count


def plot_and_save(image, labels, contours, fig_count, path):
    fig, ax = plt.subplots(figsize=(18, 12))
    plt.imshow(image)

    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], '-r', linewidth=2)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    props_main = dict(boxstyle='round', facecolor='wheat', alpha=1)

    for pair, text_label in labels:
        ax.text(pair[1], pair[0], text_label, fontsize=14,
            verticalalignment='top', bbox=props)

    main_label = f'{fig_count} cards'
    ax.text(0, 0, main_label , fontsize=24,
           verticalalignment='top', bbox=props_main)
    ax.axis('off')

    fig.savefig(path, bbox_inches='tight', transparent=True)
    print('result saved at ' + directory + '/')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Geometrica cards processing.')
    parser.add_argument('input_path',
        help='input image path (.jpg).')
    parser.add_argument('output_dir',
        nargs='?', help='output image directory.')
    args = parser.parse_args()

    input_path = args.input_path
    image = np.array(io.imread(input_path))

    if args.output_dir is not None:
        directory = args.output_dir
    else:
        directory = '.'

    out_name = 'processed_' + os.path.split(input_path)[1]
    out_path = os.path.join(directory, out_name)

    print('segmentating figures...')
    figures = get_figures(image)
    print('analyzing...')
    labels, contours, fig_count = create_labels(figures)
    plot_and_save(image, labels, contours, fig_count, out_path)
