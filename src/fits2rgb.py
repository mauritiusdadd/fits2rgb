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
import typing
import argparse
import json

import numpy as np


from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clip


COMBINE_FUNCTION_DICT = {
    'mean': np.ma.mean,
    'sum': np.ma.sum,
    'std': np.nanstd,
    'min': np.nanmin,
    'max': np.nanmax
}


DEFAULT_CONFIG = {
    'channels': {
    },
    'options': {
        "image-dir": ".",
        "out-name": "rgb",
        "out-dir": ".",
        "combine-function": "mean",
        "nsamples": 1000,
        "max-reject": 5,
        "krej": 5,
        "tile-size": 256,
        "contrast": 5,
        "gray_level": 0.3
    }
}


def __args_handler(options: typing.Optional[list] = None):
    """
    Parse cli arguments.

    Parameters
    ----------
    options : list, optional
        List of cli arguments. If it is None then arguments are read from
        sys.argv. The default is None.

    Returns
    -------
    args : argparse.Namespace
        The parsed arguments.

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


def apply_func_tiled(data, func, tile_size, *args, **kwargs):
    """
    Apply a function on a single tile.

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    func : TYPE
        DESCRIPTION.
    tile_size : TYPE
        DESCRIPTION.
    *args : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    result : TYPE
        DESCRIPTION.

    """
    data_shape = data.shape[-2:]
    if isinstance(data, np.ma.MaskedArray):
        result = np.ma.zeros(data_shape)
    else:
        result = np.zeros(data_shape)
    for j in np.arange(data_shape[0], step=tile_size):
        for k in np.arange(data_shape[1], step=tile_size):
            if len(data.shape) == 3:
                tile = data[:, j:j+tile_size, k:k+tile_size]
                processed_tile = func(tile, *args, axis=0, **kwargs).copy()
            elif len(data.shape) == 2:
                tile = data[j:j+tile_size, k:k+tile_size]
                processed_tile = func(tile, *args, **kwargs).copy()
            result[j:j+tile_size, k:k+tile_size] = processed_tile
            try:
                result[j:j+tile_size, k:k+tile_size].mask = processed_tile.mask
            except AttributeError:
                pass

    return result


def compute_on_tiles(data, func, tile_size, *args, **kwargs):
    """
    Execute a function on an image by tiling.

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    func : TYPE
        DESCRIPTION.
    tile_size : TYPE
        DESCRIPTION.
    *args : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    results : TYPE
        DESCRIPTION.

    """
    data_hieght, data_width = data.shape[-2:]
    results = []
    for j in np.arange(data_hieght, step=tile_size):
        for k in np.arange(data_width, step=tile_size):
            if len(data.shape) == 3:
                tile = data[:, j:j+tile_size, k:k+tile_size]
            elif len(data.shape) == 2:
                tile = data[j:j+tile_size, k:k+tile_size]
            try:
                results.append(func(tile, *args, **kwargs))
            except ValueError:
                continue
    return results


def process_images(filenames, combine_function, tile_size, n_samples=1000,
                   contrast=5, gray_level=0.1, max_reject=5, krej=2.5):
    """
    Process the images.

    Parameters
    ----------
    filenames : TYPE
        DESCRIPTION.
    combine_function : TYPE
        DESCRIPTION.
    tile_size : TYPE
        DESCRIPTION.
    n_samples : TYPE, optional
        DESCRIPTION. The default is 1000.
    contrast : TYPE, optional
        DESCRIPTION. The default is 5.
    gray_level : TYPE, optional
        DESCRIPTION. The default is 0.1.
    max_reject : TYPE, optional
        DESCRIPTION. The default is 5.
    krej : TYPE, optional
        DESCRIPTION. The default is 2.5.

    Returns
    -------
    rescaled : TYPE
        DESCRIPTION.

    """
    print(f"  - loading {len(filenames)} files...")

    channel_data = np.ma.masked_invalid(
        np.asarray(
            [
                fits.getdata(fname)
                for fname in filenames
            ]
        )
    )

    print("  - processing data...")
    result = apply_func_tiled(
        channel_data,
        combine_function,
        tile_size=tile_size,
    )

    print(
        f"  - log transform (contrast={contrast:.4f}; "
        f"gray={gray_level:.4f})..."
    )
    log_data = np.log10(1.0 + result - np.ma.min(result))

    subsample = np.asarray(
        compute_on_tiles(
            log_data,
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

    print(f"  - vmin={vmin:.4f}  vmax={vmax:.4f}")
    rescaled = np.clip((log_data.filled(np.nan) - vmin) / (vmax - vmin), 0, 1)

    return rescaled


def main(options: typing.Optional[list] = None):
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
    config = DEFAULT_CONFIG.copy()

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

    hdul = [fits.PrimaryHDU()]

    img_wcs = None

    for channel_name, channel_files in config['channels'].items():
        print(f"CHANNEL {channel_name}")

        filenames = [
            os.path.join(config['options']['image-dir'], x)
            for x in channel_files
        ]

        if img_wcs is None:
            img_wcs = WCS(fits.getheader(filenames[0]))

        result = process_images(
            filenames,
            combine_function=COMBINE_FUNCTION_DICT[
                config['options']['combine-function']
            ],
            tile_size=config['options']['tile-size'],
            n_samples=config['options']['nsamples'],
            krej=config['options']['krej'],
            max_reject=config['options']['max-reject'],
            contrast=config['options']['contrast'],
            gray_level=config['options']['gray_level'],
        )

        hdul.append(
            fits.ImageHDU(
                name=channel_name,
                data=result,
                header=img_wcs.to_header()
            )
        )

    hdul = fits.HDUList(hdul)
    hdul.writeto(
        os.path.join(
            config['options']['out-dir'],
            config['options']['out-name'] + '.fits'
        ),
        overwrite=True
    )
    hdul.close()
    print("DONE!")


if __name__ == '__main__':
    main()
