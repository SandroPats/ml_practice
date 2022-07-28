import os
import argparse
import math
import numpy as np
from skimage import io
from skimage.color import rgb2gray, rgb2hsv
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.filters import threshold_otsu, gaussian, sobel
from skimage import img_as_ubyte
from skimage.color import rgb2gray, rgb2hsv
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize
from skimage.morphology import medial_axis
from skan import Skeleton

class Edge:
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst
        self.val = dst - src

    def to_plot(self):
        x = [self.src[0], self.dst[0]]
        y = [self.src[1], self.dst[1]]
        return x, y

class Rectangle:
    def __init__(self, points_CCW):
        self.edges = []
        for src, dst in zip(points_CCW[:-1], points_CCW[1:]):
            self.edges.append(Edge(src, dst))
        self.edges.append(Edge(points_CCW[-1], points_CCW[0]))

    def check_point(self, points):
        inside = np.ones(points.shape[0]).astype(bool)
        for edge in self.edges:
            src_point = points - edge.src
            vec_prod = src_point[:, 1]*edge.val[0] - src_point[:, 0]*edge.val[1]
            #print(vec_prod.shape)
            inside = inside & (vec_prod > 0)

        return inside

class TwoEdgeArea:
    def __init__(self, points_CCW):
        self.edges = []
        self.edges.append(Edge(points_CCW[0], points_CCW[1]))
        self.edges.append(Edge(points_CCW[2], points_CCW[3]))

    def check_point(self, points):
        inside = np.ones(points.shape[0]).astype(bool)
        for edge in self.edges:
            src_point = points - edge.src
            vec_prod = src_point[:, 1]*edge.val[0] - src_point[:, 0]*edge.val[1]
            #print(vec_prod.shape)
            inside = inside & (vec_prod > 0)

        return inside

def clockwiseangle_and_distance(vector, refvec):
    lenvector = math.hypot(vector[0], vector[1])
    if lenvector == 0:
        return -math.pi, 0
    normalized = [vector[0]/lenvector, vector[1]/lenvector]
    dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1]
    diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]
    angle = math.atan2(diffprod, dotprod)
    if angle < 0:
        return 2*math.pi+angle
    return angle

def to_1channel(img, mode='blue'):
    if mode == 'red':
        return img[:, :, 0].copy()
    elif mode == 'green':
        return img[:, :, 1].copy()
    elif mode == 'blue':
        return img[:, :, 2].copy()
    elif mode == 'gray':
        return img_as_ubyte(rgb2gray(img))
    elif mode == 'hue':
        return img_as_ubyte(rgb2hsv(img)[:, :, 0])
    elif mode == 'saturation':
        return img_as_ubyte(rgb2hsv(img)[:, :, 1])
    elif mode == 'value':
        return img_as_ubyte(rgb2hsv(img)[:, :, 2])

class HandMarkup:
    def __init__(self):
        pass

    def _get_skeleton(self, image):
        img = to_1channel(image, 'red')
        img_sobel = sobel(img) 
        sobel_bin = img_sobel > threshold_otsu(img_sobel)
        img_bin = img > threshold_otsu(img)
        img_bin = remove_small_objects(img_bin, 30000)
        img_bin = img_bin * ~(sobel_bin)
        img_bin = (remove_small_holes((remove_small_objects(img_bin, 250))))
        self.img_bin = img_bin
        self.skeleton = skeletonize(gaussian(img_bin), method='zhang')
        skel, dists = medial_axis(img_bin, return_distance=True)
        self.dists = dists
        self.max_dist = np.max(dists)
        center = np.where(dists == self.max_dist)
        center = np.array((center[0][0], center[1][0])).reshape(-1, 2)
        self.center = center[0]

    def _prune_skel(self):
        skel = Skeleton(self.skeleton)

        endpoints_src = skel.paths.indices[skel.paths.indptr[:-1]]
        endpoints_dst = skel.paths.indices[skel.paths.indptr[1:] - 1]
        deg_src = skel.degrees[endpoints_src]
        deg_dst = skel.degrees[endpoints_dst]
        coord_src = skel.coordinates[endpoints_src].astype(int)
        coord_dst = skel.coordinates[endpoints_dst].astype(int)
        center_to_src = np.linalg.norm(coord_src - self.center, axis=1)
        center_to_dst = np.linalg.norm(coord_dst - self.center, axis=1)

        bound = 2*self.max_dist if self.max_dist < 70 else self.max_dist
        mask = ((deg_src > 1) & (deg_dst > 1) & (skel.path_lengths() < 40) &
                (center_to_src > bound) & (center_to_dst > bound))

        to_prune = np.where(mask)[0]
        skel2 = skel.prune_paths(to_prune)
        endpoints_src = skel2.paths.indices[skel2.paths.indptr[:-1]]
        endpoints_dst = skel2.paths.indices[skel2.paths.indptr[1:] - 1]
        deg_src = skel2.degrees[endpoints_src]
        deg_dst = skel2.degrees[endpoints_dst]
        mask = (skel2.path_lengths() < 40) & (deg_src == 1) & (deg_dst == 1)
        to_prune = np.where(mask)[0]
        self.skel = skel2.prune_paths(to_prune)
        endpoints_src = self.skel.paths.indices[self.skel.paths.indptr[:-1]]
        endpoints_dst = self.skel.paths.indices[self.skel.paths.indptr[1:] - 1]
        deg_src = self.skel.degrees[endpoints_src]
        deg_dst = self.skel.degrees[endpoints_dst]
        self.finger_inds = np.where((deg_src != -1))[0]

    def validate_fingers(self):
        fingers_data = []
        for i in self.finger_inds:
            path = self.skel.path_coordinates(i)
            path = path.astype(int)
            start = path[0]
            end = path[-1]
            c_to_end = np.linalg.norm(self.center - end)
            c_to_start = np.linalg.norm(self.center - start)
            state = c_to_end > c_to_start
            if state:
                top = end
                top_idx = path.shape[0] - 1
                root = start
            else:
                top = start
                top_idx = 0
                root = end
            top_rad = self.dists[top[0], top[1]]
            path_rad = self.dists[path[:, 0], path[:, 1]]
            max_rad = path_rad.max()
            diff = np.diff(path_rad, prepend=0)
            from_center = np.linalg.norm(path - self.center, axis=1)
            inds_diff = abs(np.arange(path.shape[0]) - top_idx)
            cond = ((diff < 0) & (path_rad < 0.3 * self.max_dist) &
                    (path_rad > top_rad) & (from_center > 1.2*self.max_dist) &
                    (inds_diff < 150))
            bot_cands = np.where(cond)[0]
            if bot_cands.size != 0:
                bot_idx = bot_cands[0] if state else bot_cands[-1]
                bot = path[bot_idx]
            else:
                bot = root
                bot_idx = 0 if state else -1
            max_rad = path_rad[bot_idx:].max() if state else path_rad[:bot_idx].max()
            if max_rad > 35:
                continue
            path = path[bot_idx:] if state else path[:bot_idx]
            fingers_data.append([path, top, bot])

        func = (lambda val: val[0].shape[0]**0.5 *
                np.linalg.norm(val[1] - self.center)/
                np.linalg.norm(val[2] - self.center))
        fingers = sorted(fingers_data, key=func)[-5:]
        mean = sum(map(lambda val: np.mean(val[0], axis=0), fingers)) / 5
        func = lambda val: clockwiseangle_and_distance(
            val[1] - self.center,
            self.center - mean
        )
        self.fingers = sorted(fingers, key=func)

    def get_pos(self, img, baseline_coefs=np.array([0.7,0.65,0.68,0.65])):
        self._get_skeleton(img)
        self._prune_skel()
        self.validate_fingers()
        tops = np.array([val[1] for val in self.fingers])
        bots = np.array([val[2] for val in self.fingers])
        vecs = tops - bots
        inds = np.indices(self.img_bin.shape)
        pic_inds = np.stack((inds[0][:, :, None],
                             inds[1][:, :, None]), axis=3).reshape(-1, 2)

        coefs = []
        self.recs = []
        gen = zip(tops[:-1], bots[:-1], tops[1:], bots[1:])
        for top_l, bot_l, top_r, bot_r in gen:
            rec = Rectangle([bot_r, top_r, top_l, bot_l])
            inside = rec.check_point(pic_inds)
            ins = pic_inds[np.where(inside==True)[0], :]
            rec_pix = self.img_bin[ins[:, 0], ins[:, 1]]
            if rec_pix.shape[0] == 0:
                coef = 0
            else:
                coef = rec_pix.sum() /rec_pix.shape[0]
            self.recs.append(rec)
            coefs.append(coef)

        signs='5'
        for i, coef in enumerate(coefs[:], 2):
            #print(signs)
            if coef >= baseline_coefs[i-2]:
                signs = str(6-i) + '+' + signs
            else:
                signs = str(6-i) + '-' + signs

        self.tops = tops
        self.bots = bots
        self.vecs = vecs

        return signs

    def find_line(self):
        self.border = sobel(self.img_bin) > 0
        border_points = np.stack(np.where(self.border), axis=1)
        tips = []
        for top, vec in zip(self.tops, self.vecs):
            if vec[0] == 0:
                t_vals = abs(np.diff((border_points[:, 1] - top[1])/(vec[1])))
            elif vec[1] == 0:
                t_vals = abs(np.diff((border_points[:, 0] - top[0])/(vec[0])))
            else:
                t_vals = abs(np.diff((border_points - top)/(vec), axis=1))
            idxs = np.where(t_vals < 0.12)[0]
            idx = np.argmin(np.linalg.norm(border_points[idxs] - top, axis=1))
            tip = border_points[idxs][idx]
            tips.append(tip)
        valleys = []
        areas =[]
        gen = zip(self.tops[:-1], self.bots[:-1], self.tops[1:], self.bots[1:])
        for i, (top_l, bot_l, top_r, bot_r) in enumerate(gen):
            area = TwoEdgeArea([bot_r, top_r, top_l, bot_l])
            inside = area.check_point(border_points)
            valley2 = (bot_r+bot_l)/2
            ins = border_points[np.where(inside==True)[0], :]
            ins = ins[np.linalg.norm(ins - valley2, axis=1) < 70]
            if ins.size == 0:
                valleys.append(valley2)
                continue
            valley= ins[np.linalg.norm(ins - self.center, axis=1).argmin()]
            alpha = 0.7
            if i == 3:
                valley = alpha*valley + (1-alpha)*bot_r
            else:
                valley = alpha*valley + (1-alpha)*valley2
            valleys.append(valley)

        self.valleys = valleys[::-1]
        self.tips = tips[::-1]
        line = []
        for valley, tip in list(zip(valleys, tips)):
            line.append(tip)
            line.append(valley)
        line.append(tips[-1])

        return np.array(line[::-1])

def plot_and_save(image, pos, line, path):
    plt.figure(figsize=(10, 8))
    props = dict(boxstyle='round', facecolor='wheat', alpha=1)
    plt.imshow(image)
    plt.plot(line[:, 1], line[:, 0], color='orange')
    plt.text(0,0, pos, bbox=props, fontsize=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    print('result saved at ' + directory)
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
    res_path = os.path.join(directory, 'Resluts.txt')
    tail = os.path.split(input_path)[-1]

    print('processing input image...')
    hm = HandMarkup()
    pos = hm.get_pos(image)
    line = hm.find_line()
    tips = hm.tips
    valleys = hm.valleys
    line_out = '!,' + tail + ','
    for tip in tips:
        x = int(tip[1])
        y = int(tip[0])
        coord = 'T {} {},'.format(x, y)
        line_out += coord
    for valley in valleys:
        x = int(valley[1])
        y = int(valley[0])
        coord = 'V {} {},'.format(x, y)
        line_out += coord
    line_out += '?'
    with open(res_path, 'w') as file:
        file.write(pos + '\n')
        file.write(line_out + '\n')
    plot_and_save(image, pos, line, out_path)
