#!/usr/bin/env python
"""
FITS2RGB.

Combine multiple FITS images into an RGB FITS image.

Copyright (C) 2022  Maurizio D'Addona <mauritiusdadd@gmail.com>

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import os
import sys
import argparse
import json

from typing import Tuple, List, Dict, Union, Any, Optional, Callable

import numpy as np

from reproject import reproject_interp, reproject_exact
from reproject.mosaicking import find_optimal_celestial_wcs

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.stats import sigma_clip
from astropy import units


COMBINE_FUNCTION_DICT: Dict[str, Callable] = {
    'mean': np.nanmean,
    'sum': np.nansum,
    'std': np.nanstd,
    'min': np.nanmin,
    'max': np.nanmax
}


DEFAULT_CONFIG: Dict[str, Any] = {
    'channels': {
    },
    'options': {
        "image-dir": ".",
        "out-name": "rgb",
        "out-dir": ".",
        "combine-function": "mean",
        "reproj-method": "exact",
        "nsamples": 1000,
        "max-reject": 5,
        "krej": 5,
        "tile-size": 256,
        "contrast": 5,
        "gray-level": 0.3,
        "color-scale": "log",
        "out-shape": None,
        "out-dtype": 'int16'
    }
}


def __args_handler(options: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse cli arguments.

    :param options: If it is None then arguments are read from sys.argv.
        The default is None.
    :return: The parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config-file', '-c', type=str, default='fits2rgb.json',
        metavar='CONFIG_FILE', help='Set the configuration to read.'
    )

    parser.add_argument(
        '--dump-defaults', '-d', action='store_true', default=False,
        help='Save a sample default configuration to a file named '
        'fits2rgb.json in the current working directory and exit.'
    )

    if options is not None:
        args = parser.parse_args(options)
    else:
        args = parser.parse_args()

    return args


def apply_func_tiled(
    data: np.ndarray,
    func: Callable,
    tile_size: int,
    *args,
    **kwargs
) -> np.ndarray:
    """
    Apply a function on the input data in tiles.
    :param data: The input data.
    :param func: The function to be applied.
    :param tile_size: The size of the tile.
    :param args: PLACEHOLDER, NOT USED.
    :param kwargs: PLACEHOLDER, NOT USED.
    :return: func(data)
    """
    if (len(data.shape) < 2) or (len(data.shape) > 3):
        raise ValueError("Only 2D and 3D array are supported!")

    data_shape: Tuple[int, ...] = data.shape[-2:]

    if isinstance(data, np.ma.MaskedArray):
        result = np.ma.zeros(data_shape)
    else:
        result = np.zeros(data_shape)

    for j in np.arange(start=0, stop=data_shape[0], step=tile_size):
        for k in np.arange(start=0, stop=data_shape[1], step=tile_size):
            if len(data.shape) == 3:
                tile = data[:, j:j+tile_size, k:k+tile_size]
                processed_tile = func(tile, *args, axis=0, **kwargs).copy()
            else:
                tile = data[j:j+tile_size, k:k+tile_size]
                processed_tile = func(tile, *args, **kwargs).copy()
            result[j:j+tile_size, k:k+tile_size] = processed_tile

            try:
                result[j:j+tile_size, k:k+tile_size].mask = processed_tile.mask
            except AttributeError:
                pass

    return result


def compute_on_tiles(
    data: np.ndarray,
    func: Callable,
    tile_size: int,
    *args,
    **kwargs
) -> List[Any]:
    """
    Execute a function on an image by tiling.

    :param data: The input data.
    :param func: The function to be applied.
    :param tile_size: The size of the tile.
    :param args: PLACEHOLDER, NOT USED.
    :param kwargs: PLACEHOLDER, NOT USED.
    :return: A list of values
    """
    if (len(data.shape) < 2) or (len(data.shape) > 3):
        raise ValueError("Only 2D and 3D array are supported!")

    data_shape: Tuple[int, ...] = data.shape[-2:]
    results = []

    for j in np.arange(start=0, stop=data_shape[0], step=tile_size):
        for k in np.arange(start=0, stop=data_shape[1], step=tile_size):
            if len(data.shape) == 3:
                tile = data[:, j:j+tile_size, k:k+tile_size]
            else:
                tile = data[j:j+tile_size, k:k+tile_size]
            try:
                results.append(func(tile, *args, **kwargs))
            except ValueError:
                continue
    return results


def get_best_wcs(
    hdu_list: Union[
        List[Union[fits.PrimaryHDU, fits.ImageHDU]],
        List[Tuple[np.ndarray, fits.Header]],
        List[Tuple[np.ndarray, WCS]],
    ],
    target_shape: Optional[Tuple[int, int]] = None
) -> Tuple[WCS, Tuple[int, int]]:
    """
    Get the best WCS for the input HDUs, compatible with the desired out shape.
    :param hdu_list: A list of HDUs, (array, WCS) or (array, header) tuples.
    :param target_shape: Optional, the desired width and height.
    :return: The best WCS and the output shape.
    """
    # Get a first guess of the best WCS compatible with the input HDUs
    first_guess_wcs, first_guess_out_shape = find_optimal_celestial_wcs(
        hdu_list
    )

    if target_shape is None:
        return first_guess_wcs, first_guess_out_shape

    # Get the pixel scale of the first-guess WCS
    first_guess_pixel_scale = proj_plane_pixel_scales(
        first_guess_wcs
    ) * first_guess_wcs.wcs.cunit

    # Define a new pixel scale compatible with the desired output shape
    scale_parameter = np.array(first_guess_out_shape) / np.array(target_shape)
    target_resolution = np.min(first_guess_pixel_scale * scale_parameter)

    # Recompute the best WCS with the desired pixel scale
    out_wcs, out_shape = find_optimal_celestial_wcs(
        hdu_list,
        resolution=target_resolution
    )

    return out_wcs, out_shape


def process_images(
    hdu_list: List[Union[fits.PrimaryHDU, fits.ImageHDU]],
    out_wcs: WCS,
    out_shape: Tuple[int, int],
    combine_function: Callable,
    reproj_method: str = 'exact',
    tile_size: int = 256,
    n_samples: int = 1000,
    contrast: float = 5.0,
    gray_level: float = 0.1,
    color_scale: str = "log",
    max_reject: float = 5,
    krej: float = 2.5,
) -> np.ndarray:
    """
    Process the images.

    :param hdu_list:
    :param out_wcs:
    :param out_shape:
    :param combine_function:
    :param reproj_method:
    :param tile_size:
    :param n_samples:
    :param contrast:
    :param gray_level:
    :param color_scale:
    :param max_reject:
    :param krej:
    :return:
    """
    if reproj_method == 'exact':
        reproj_func = reproject_exact
    elif reproj_method == 'interp':
        reproj_func = reproject_interp
    else:
        reproj_func = None

    rebinned_list = []
    rebinned_mask_list = []

    print("  - reprojecting images", end='')
    for a_hdu in hdu_list:
        sys.stdout.write('.')
        sys.stdout.flush()
        if reproj_func is not None:
            rebinned, rebinned_mask = reproj_func(
                a_hdu, out_wcs, shape_out=out_shape
            )
            rebinned_list.append(rebinned)
            rebinned_mask_list.append(rebinned_mask)
        else:
            rebinned_list.append(a_hdu.data)
            rebinned_mask_list.append(np.isfinite(a_hdu.data))
    print("")

    channel_data = np.ma.masked_invalid(
        np.asarray(rebinned_list)
    )

    channel_mask = np.ma.masked_invalid(
        np.asarray(rebinned_mask_list)
    )

    print("  - processing data...")
    result = apply_func_tiled(
        channel_data,
        combine_function,
        tile_size=tile_size,
    )

    result_mask = apply_func_tiled(
        channel_mask,
        combine_function,
        tile_size=tile_size,
    )

    print(
        f"  - log transform (contrast={contrast:.4f}; "
        f"gray={gray_level:.4f})..."
    )

    color_scale = color_scale.lower()
    if color_scale == 'log':
        img_data = np.log10(1.0 + result - np.ma.min(result))
    else:
        img_data = result

    subsample = np.asarray(
        compute_on_tiles(
            img_data,
            lambda x: np.random.choice(
                np.ravel(x[~x.mask]),
                size=10
            ),
            tile_size
        )
    ).flatten()

    clipped_subsample = sigma_clip(
        subsample,
        sigma=max_reject,
        maxiters=krej
    )[:n_samples]

    median_val = np.ma.median(clipped_subsample)
    std_val = np.ma.std(clipped_subsample)
    vmin = median_val - contrast*gray_level*std_val
    vmax = median_val + contrast*(1 - gray_level)*std_val

    print(f"  - median={median_val:.4f}  std.dev.={std_val:.4f}")
    print(f"  - vmin={vmin:.4f}  vmax={vmax:.4f}")
    rescaled = np.clip(
        (img_data.filled(np.nan) - vmin) / (vmax - vmin),
        0,
        1
    )

    return rescaled, result_mask


def main(options: Optional[List[str]] = None) -> None:
    """
    Run the main program.

    Parameters
    ----------
    options : list, optional
        List of cli arguments. If it is None then arguments are read from
        sys.argv. The default is None.

    Returns
    -------
    None.

    """
    args = __args_handler(options)
    config: Dict[str, Any] = DEFAULT_CONFIG.copy()

    if args.dump_defaults:
        config["channels"]['R'] = []
        config["channels"]['G'] = []
        config["channels"]['B'] = []
        try:
            with open("fits2rgb.json", 'w') as f:
                json.dump(config, f, indent=4)
        except Exception:
            print("Error writing to file fits2rgb.json")
            sys.exit(1)
        sys.exit(0)

    try:
        with open(args.config_file, 'r') as f:
            for section_name, section_dict in json.load(f).items():
                config[section_name].update(section_dict)
    except (OSError, IOError, FileNotFoundError) as exc:
        print(f"Error opening config file {args.config_file}: {exc}")
        sys.exit(1)

    out_shape: Union[None, Tuple[int, int]]
    try:
        out_shape = config['options']['out-shape']
    except KeyError:
        out_shape = None

    hdu_list_data: List[Union[fits.PrimaryHDU, fits.ImageHDU]] = [
        fits.PrimaryHDU()
    ]
    hdu_list_mask: List[Union[fits.PrimaryHDU, fits.ImageHDU]] = [
        fits.PrimaryHDU()
    ]

    # Load all files to get the best WCS and output shape
    open_hdul: List[fits.HDUList] = []
    channels_dict = {}

    try:
        for channel_name, channel_files in config['channels'].items():
            print(f"Loading files for channel {channel_name}")
            hlist = []
            for filename in channel_files:
                hl = fits.open(
                    os.path.join(config['options']['image-dir'], filename),
                    do_not_scale_image_data=True
                )
                open_hdul.append(hl)
                hlist.append(hl[0])
            channels_dict[channel_name] = hlist

        all_hdus = [
            hdu for hlist in channels_dict.values() for hdu in hlist
        ]

        # If we don't want to reproject, we must assure that all the images
        # have the same shape. We assume that they have been already registered
        if config['options']['reproj-method'] is None:
            best_shape = None
            best_wcs = None
            for hdu in all_hdus:
                if best_shape is None:
                    best_shape = hdu.data.shape
                    best_wcs = WCS(hdu.header)
                elif out_shape == hdu.data.shape:
                    raise ValueError(
                        "Images have different shapes and no reprojection "
                        "method is used!"
                    )
        else:
            print(f"Target shape: {out_shape}")
            best_wcs, best_shape = get_best_wcs(
                hdu_list=all_hdus,
                target_shape=out_shape
            )

        print(f"Output shape: {best_shape}")

        for channel_name, channel_hdu_list in channels_dict.items():
            print(f"CHANNEL {channel_name}")

            result, result_mask = process_images(
                channel_hdu_list,
                out_wcs=best_wcs,
                out_shape=best_shape,
                combine_function=COMBINE_FUNCTION_DICT[
                    config['options']['combine-function']
                ],
                reproj_method=config['options']['reproj-method'],
                tile_size=config['options']['tile-size'],
                n_samples=config['options']['nsamples'],
                krej=config['options']['krej'],
                color_scale=config['options']['color-scale'],
                max_reject=config['options']['max-reject'],
                contrast=config['options']['contrast'],
                gray_level=config['options']['gray-level'],
            )

            if isinstance(result, np.ma.core.masked_array):
                result = result.filled(fill_value=np.nan)

            if isinstance(result_mask, np.ma.core.masked_array):
                result_mask = result_mask.filled(fill_value=np.nan)

            try:
                out_dtype = config['options']['out-dtype']
            except KeyError:
                out_dtype = 'float16'
            else:
                out_dtype = out_dtype.lower()

            if out_dtype.startswith('int'):
                max_val = np.iinfo(np.dtype(out_dtype)).max
                result = result - np.nanmin(result)
                result = result * max_val / np.nanmax(result)

                result_mask = result_mask - np.nanmin(result_mask)
                result_mask = result_mask * max_val / np.nanmax(result_mask)

            channel_hdu = fits.ImageHDU(
                name=channel_name,
                data=result,
                header=best_wcs.to_header()
            )

            channel_mask_hdu = fits.ImageHDU(
                name=channel_name,
                data=result_mask,
                header=best_wcs.to_header()
            )

            if out_dtype.startswith('int'):
                channel_hdu.scale(out_dtype)
                channel_mask_hdu.scale(out_dtype)

            hdu_list_data.append(channel_hdu)
            hdu_list_mask.append(channel_mask_hdu)

        hdul: fits.HDUList = fits.HDUList(hdu_list_data)
        hdul.writeto(
            os.path.join(
                config['options']['out-dir'],
                config['options']['out-name'] + '.fits'
            ),
            overwrite=True
        )
        hdul.close()

        hdul: fits.HDUList = fits.HDUList(hdu_list_mask)
        hdul.writeto(
            os.path.join(
                config['options']['out-dir'],
                config['options']['out-name'] + '_mask.fits'
            ),
            overwrite=True
        )
        hdul.close()
        print("DONE!")
    except Exception as exc:
        print(f"An error has occured: {str(exc)}")
    finally:
        for hdul in open_hdul:
            hdul.close()


if __name__ == '__main__':
    main()
