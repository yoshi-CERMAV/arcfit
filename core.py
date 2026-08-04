import numpy as np
import pyFAI
from PIL import Image
import fabio
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.optimize import least_squares, root_scalar
from scipy.interpolate import SmoothBivariateSpline, bisplrep, bisplev
from .utils import gaussian2d, dgaussian2d, func, jacobian

class ArcFit:
    def __init__(self):
        print(
            "Need to read poni file (load_poni), load data (load_tiff), and mask (load_mask_edf). "
            "Then create grids with 'create_grids(nq_bins, q_step, nChi_bins)'"
        )

    def load_poni(self, poni_file):
        self.ai = pyFAI.load(poni_file)
        self.chi_array = self.ai.chiArray().ravel()
        self.q_array = self.ai.qArray().ravel()
        self.vsize, self.hsize = self.ai.detector.shape

    def load_tiff(self, file_path):
        im = Image.open(file_path)
        self.data = np.array(im)
        self.mean = self.data.mean()
        self.std = self.data.std()

    def correct_border(self, selection, ratio):
        self.data.ravel()[selection] *= ratio

    def load_mask_edf(self, mask_file):
        self.mask = fabio.open(mask_file).data.ravel()

    def create_polygon_mask(self, polygon_vertices, show=True):
        x, y = np.meshgrid(np.arange(self.hsize), np.arange(self.vsize))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x, y)).T
        path = Path(polygon_vertices)
        mask = path.contains_points(points)
        mask = mask.reshape(self.vsize, self.hsize)
        if show:
            plt.imshow(mask, vmax=1, vmin=0)
        return mask

    def mask_polygon(self, a):
        self.mask1 = self.create_polygon_mask(self.hsize, self.vsize, a)

    def add_mask(self, mask):
        self.mask += mask

    def create_grids(self, nq_bins, q_step, nChi_bins):
        self.nq_bins = nq_bins
        self.q_step = q_step
        self.r_q_step = 1.0 / q_step
        self.nChi_bins = nChi_bins
        self.r_chi_step = 0.5 * nChi_bins / np.pi
        grid = [[[] for _ in range(self.nq_bins)] for _ in range(self.nChi_bins)]

        for idx, (Chii, qi) in enumerate(zip(self.chi_array, self.q_array)):
            if self.mask[idx]:
                continue
            Chi_bin = min(int((Chii + np.pi) * self.r_chi_step), nChi_bins - 1)
            q_bin = min(int(qi * self.r_q_step), self.nq_bins - 1)
            grid[Chi_bin][q_bin].append(int(idx))
        self.grid_array = np.array(grid, dtype=object)

    def set_chirange(self, chi_range):
        self.chi_range = chi_range
        self.ichi_range = int(chi_range * self.r_chi_step) + 1

    def set_qrange(self, q_range):
        self.q_range = q_range
        self.iq_range = int(q_range * self.r_q_step) + 1

    def print_results(self):
        print("Result:", self.height, self.xpos, self.ypos, self.xwidth, self.ywidth)
        print("Integral:", self.height * self.xwidth * self.ywidth)

    def plot_selected(self, axe):
        axe.scatter(self.chi1, self.q1, c=self.target, marker=".")

    def select_pixels(self, chi, q):
        self.q = q
        self.chi = chi % (np.pi * 2)
        qi = q * self.r_q_step
        chii = self.chi * self.r_chi_step
        c0 = int(np.floor((self.chi - self.chi_range) * self.r_chi_step))
        c1 = int((self.chi + self.chi_range) * self.r_chi_step + 1)
        q0 = max(0, int((q - self.q_range) * self.r_q_step))
        q1 = min(self.nq_bins, int((q + self.q_range) * self.r_q_step) + 1)

        if c0 >= 0 and c1 < self.nChi_bins:
            self.selection = np.concatenate(
                np.concatenate(self.grid_array[c0:c1, q0:q1])
            ).ravel().astype(int)
        elif c0 < 0:
            t = np.concatenate(np.concatenate(self.grid_array[0:c1, q0:q1]))
            t1 = np.concatenate(np.concatenate(self.grid_array[self.nChi_bins + c0 : self.nChi_bins, q0:q1]))
            self.selection = np.concatenate([t, t1]).ravel().astype(int)
        else:
            t = np.concatenate(np.concatenate(self.grid_array[0 : c1 - self.nChi_bins + 1, q0:q1]))
            t1 = np.concatenate(np.concatenate(self.grid_array[c0 : self.nChi_bins, q0:q1]))
            self.selection = np.concatenate([t, t1]).ravel().astype(int)
        return self.selection

    def modulo(self, x):
        return (x + np.pi) % (np.pi * 2) - np.pi

    def fit_linear(self):
        a = self.selection
        temp = np.copy(self.chi_array[a])
        self.chi1 = self.modulo(temp + np.pi - self.chi)
        self.q1 = self.q_array[a]
        self.target = self.data.ravel()[a]
        A = np.array(
            [
                gaussian2d(self.chi1, self.q1, 1, 0, self.q, self.xwidth, self.ywidth),
                self.q1,
                np.ones(self.q1.size),
            ]
        ).transpose()
        res = lstsq(A, self.target)
        self.height = res[0][0]
        self.const = res[0][1]
        self.slope = res[0][2]

    def fit_non_linear(self):
        var = np.array([self.height, 0, self.q, self.xwidth, self.ywidth, self.const, self.slope])
        self.res = least_squares(func, var, jac=jacobian, method="lm", args=[self.chi1, self.q1, self.target])
        self.height = self.res.x[0]
        self.xpos = self.res.x[1]
        self.ypos = self.res.x[2]
        self.xwidth = self.res.x[3]
        self.ywidth = self.res.x[4]
        self.const = self.res.x[5]
        self.slope = self.res.x[6]

    def smooth_and_fit(self, chi_step=0.0002, q_step=0.001, m=100):
        s = m + np.sqrt(2 * m)
        a = self.selection
        self.chi1 = self.modulo(np.copy(self.chi_array[a]) + np.pi - self.chi)
        self.q1 = self.q_array[a]
        self.target = self.data.ravel()[a]
        minValue = self.target.min()
        maxValue = self.target.max()
        meanValue = self.target.mean()
        stdDev = self.target.std()
        plt.scatter(
            self.chi1,
            self.q1,
            c=self.data.ravel()[a],
            marker=".",
            vmin=meanValue - stdDev,
            vmax=meanValue + 2 * stdDev,
            s=10,
        )
        plt.show()

        chi0 = -self.chi_range
        q0 = self.q - self.q_range
        q1 = self.q + self.q_range
        tx = np.linspace(-self.chi_range, self.chi_range, 12)
        ty = np.linspace(q0, q1, 12)
        tck = bisplrep(
            self.chi1, self.q1, self.data.ravel()[a], task=-1, kx=3, ky=3, tx=tx, ty=ty, s=s
        )
        test_x = np.arange(chi0, self.chi_range, chi_step)
        test_y = np.arange(q0, self.q + self.q_range, q_step)
        smth_result = bisplev(test_x, test_y, tck).T
        plt.imshow(smth_result)
        height = smth_result.max()
        base = smth_result.min()
        mid_point = (height + base) * 0.5
        x0 = smth_result.argmax() % smth_result.shape[1] * chi_step + chi0
        y0 = smth_result.argmax() / smth_result.shape[1] * q_step + q0

        def equation(x):
            return bisplev(x0, x, tck) - mid_point

        def equation1(x):
            return bisplev(x, y0, tck) - mid_point

        solution0 = root_scalar(equation, bracket=[y0, y0 + self.q_range]).root
        solution1 = root_scalar(equation, bracket=[q0, y0]).root
        width_y = solution0 - solution1
        solution0 = root_scalar(equation1, bracket=[x0, x0 + self.chi_range]).root
        solution1 = root_scalar(equation1, bracket=[chi0, x0]).root
        width_x = solution0 - solution1
        self.var = np.array([height, x0, y0, width_x, width_y, base, 0])
        self.res = least_squares(
            func, self.var, jac=jacobian, method="lm", args=[self.chi1, self.q1, self.data.ravel()[a]]
        )
        self.height = self.res.x[0]
        self.xpos = self.res.x[1]
        self.ypos = self.res.x[2]
        self.xwidth = self.res.x[3]
        self.ywidth = self.res.x[4]
        self.const = self.res.x[5]
        self.slope = self.res.x[6]
        return self.res

    def pix2coord(self, x, y):
        pos = y * self.hsize + x
        return self.chi_array[pos], self.q_array[pos]

    def fit_peak_at(self, x, y):
        self.fit_peak_linear_at(x, y)
        self.fit_non_linear()
        self.print_results()

    def fit_peak_linear_at(self, x, y):
        chi, q = self.pix2coord(x, y)
        self.select_pixels(chi, q)
        self.fit_linear()

    def fit_peak_at_polar(self, chi, q):
        self.select_pixels(chi, q)
        self.fit_linear()
        self.fit_non_linear()
        self.print_results()

    def plot_res_diff(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(self.chi1, self.q1, c=self.target, marker=".")
        axes[1].scatter(self.chi1, self.q1, c=self.res.fun, marker=".")
