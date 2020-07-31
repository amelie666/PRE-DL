from pre_dl.evaluator.colormaps import RainGaugeColorMap, RainGaugeDiffColorMap, RainColorMap, RainDiffColorMap

import numpy as np
import rasterio as rio
import geopandas as gpd
from cartopy.feature import ShapelyFeature
from matplotlib import pyplot as plt


class PreDraw(object):

    def __init__(self, crs, extent, res):
        super(PreDraw, self).__init__()
        self.crs = crs
        self.extent = extent
        self.res = res
        self.shape = (int((self.extent[1] - self.extent[0]) / res),
                      int((self.extent[3] - self.extent[2]) / res))

    def plot(self):
        fig = plt.figure(figsize=[8, 8])
        ax = fig.add_subplot(1, 1, 1,
                             projection=self.crs)
        ax.set_extent(self.extent, self.crs)
        return fig, ax

    def draw_dem(self, dem_tif, fig, main_ax, **kwargs):
        with rio.open(dem_tif) as f:
            img = f.read(1)
        ax_img = main_ax.imshow(img,
                                transform=self.crs,
                                extent=(f.bounds.left, f.bounds.right, f.bounds.bottom, f.bounds.top),
                                zorder=2,
                                origin='upper',
                                **kwargs)
        color_ax = fig.add_axes([0.08, 0.15, 0.01, 0.7])
        fig.colorbar(ax_img, color_ax)
        color_ax.yaxis.set_ticks_position('left')
        color_ax.tick_params(labelsize=8)
        return main_ax

    def draw_shape_file(self, shp_file, ax, **kwargs):
        gdf = gpd.read_file(shp_file)
        sf = ShapelyFeature(gdf['geometry'], self.crs)
        ax.add_feature(sf, **kwargs)
        return ax

    def draw_rain_gauge(self, rain_location_tif, fig, main_ax, **kwargs):
        cmc = RainGaugeColorMap()
        kwargs.update({'cmap': cmc.cmap, 'vmax': cmc.vmax, 'vmin': cmc.vmin})
        with rio.open(rain_location_tif) as f:
            img = f.read(1)
        ax_img = main_ax.imshow(img,
                                transform=self.crs,
                                extent=(f.bounds.left, f.bounds.right, f.bounds.bottom, f.bounds.top),
                                zorder=2,
                                origin='upper',
                                **kwargs)
        color_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
        color_ax = cmc.plot(color_ax)
        color_ax.tick_params(labelsize=8)
        main_ax.set_title('Rain Gauge')
        return main_ax

    def draw_precipitation(self, pre_tif, fig, main_ax, **kwargs):
        color_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
        cmc = RainColorMap()
        color_ax = cmc.plot(color_ax)
        kwargs.update({'cmap': cmc.cmap, 'vmax': cmc.vmax, 'vmin': cmc.vmin})
        with rio.open(pre_tif) as f:
            img = f.read(1)
        ax_img = main_ax.imshow(img,
                                transform=self.crs,
                                extent=(f.bounds.left, f.bounds.right, f.bounds.bottom, f.bounds.top),
                                zorder=2,
                                origin='upper',
                                **kwargs)
        color_ax.tick_params(labelsize=8)
        main_ax.set_title('Precipitation')
        return main_ax

    def evaluate_rain_gauge(self, true_rain_tif,
                            true_location_tif,
                            pred_location_tif,
                            fig, main_ax, **kwargs):
        color_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
        cmc = RainGaugeDiffColorMap()
        color_ax = cmc.plot(color_ax)
        kwargs.update({'cmap': cmc.cmap, 'vmax': cmc.vmax, 'vmin': cmc.vmin})
        with rio.open(true_location_tif) as f:
            l_y = f.read(1)
        with rio.open(true_rain_tif) as f:
            t_y = f.read(1)
            t_y[t_y > 0] = 1
        with rio.open(pred_location_tif) as f:
            p_y = f.read(1)
            threshold = kwargs.get('threshold', 0.5)
            p_y[p_y > threshold] = 1
            p_y[p_y <= threshold] = 0
        diff = np.zeros_like(t_y, dtype=np.uint8)
        diff[np.logical_and(t_y == 0, p_y == 0)] = 1
        diff[np.logical_and(t_y == 1, p_y == 0)] = 2
        diff[np.logical_and(t_y == 0, p_y == 1)] = 3
        diff[np.logical_and(t_y == 1, p_y == 1)] = 4
        diff[np.logical_and(diff == 1, l_y == 0)] = 0  # set no rain gauge
        ax_img = main_ax.imshow(diff,
                                transform=self.crs,
                                extent=(f.bounds.left, f.bounds.right, f.bounds.bottom, f.bounds.top),
                                zorder=2,
                                origin='upper',
                                **kwargs)
        color_ax.tick_params(labelsize=8)
        main_ax.set_title('Rain Gauge Diff')
        return main_ax

    def evaluate_precipitation(self, true_rain_tif,
                               prediction_rain_tif,
                               fig, main_ax, **kwargs):
        color_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
        cmc = RainDiffColorMap()
        color_ax = cmc.plot(color_ax)
        kwargs.update({'cmap': cmc.cmap, 'vmax': cmc.vmax, 'vmin': cmc.vmin})
        with rio.open(true_rain_tif) as f:
            t_y = f.read(1)
        with rio.open(prediction_rain_tif) as f:
            p_y = f.read(1)
        diff = np.abs(t_y - p_y)
        no_rain_idx = np.logical_and(t_y == 0, p_y == 0)
        diff[no_rain_idx] = -1
        ax_img = main_ax.imshow(diff,
                                transform=self.crs,
                                extent=(f.bounds.left, f.bounds.right, f.bounds.bottom, f.bounds.top),
                                zorder=2,
                                origin='upper',
                                **kwargs)
        color_ax.tick_params(labelsize=8)
        main_ax.set_title('Rain Gauge Diff')
        return main_ax

    def decorate(self, ax):
        ax.gridlines()
