#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fits2rgb

Make RGB from single fits images.

Copyright (C) 2022-2024  Maurizio D'Addona <mauritiusdadd@gmail.com>
"""
import os
import sys
import json
import time
import pathlib
import zipfile

from typing import List
from urllib import request

from tqdm.auto import tqdm
import unittest

import numpy as np
from astropy.io import fits

import fits2rgb.core

# Sample files from https://esahubble.org/projects/fits_liberator/m31data/
TEST_FILES = [
    "https://esahubble.org/static/projects/fits_liberator/datasets/m31/f001a066.zip",
    "https://esahubble.org/static/projects/fits_liberator/datasets/m31/f001a1hr.zip",
    "https://esahubble.org/static/projects/fits_liberator/datasets/m31/f001a25o.zip",
]

TEST_DATA_PATH = os.path.join(pathlib.Path(__file__).parent.resolve(), "data")

def download_files(
    url_list: List[str],
    out_dir: str = TEST_DATA_PATH,
    use_cached: bool = True,
    report_interval: int = 1
) -> List[str]:
    """
    Download a set of files with a progressbar.

    :param url_list:
    :param out_dir:
    :param use_cached:
    :param report_interval:
    :return:
    """

    # Thanks to Lei Mao
    # https://gist.github.com/leimao/37ff6e990b3226c2c9670a2cd1e4a6f5
    def report_hook(t, b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update((b - report_hook.last_b) * bsize)
        report_hook.last_b = b
    report_hook.last_b = 0

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    outfile_list = []
    for target_url in url_list:
        target_file = os.path.basename(target_url)
        out_file = os.path.join(out_dir, target_file)
        if not (use_cached and os.path.isfile(out_file)):
            print(f"Downloading {target_file}...")
            try:
                with tqdm(
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                    miniters=1,
                    desc=out_file
                ) as t:
                    package_out_file, headers = request.urlretrieve(
                        target_url,
                        out_file,
                        reporthook=report_hook(t)
                    )
            except Exception as exc:
                print(
                    f"An exception occurred while downloading {target_file}:",
                    str(exc),
                    file=sys.stderr
                )
                continue
        else:
            print(f"Using cached {target_file}...")
        outfile_list.append(os.path.realpath(out_file))
    return outfile_list

def download_test_data() -> None:
    zip_files = download_files(
        url_list=TEST_FILES,
        out_dir=TEST_DATA_PATH,
        use_cached=True
    )

    for file_path in zip_files:
        with zipfile.ZipFile(file_path) as zf:
            zf.extractall(path=TEST_DATA_PATH)

class TestFits2Rgb(unittest.TestCase):

    def test_rgb_log(self, cfg_fname="test_log_cfg.json"):
        print("***Testing m31, log scale, interpolation***")
        download_test_data()

        channels = {
            'R': ['f001a25o.fits', ],
            'G': ['f001a1hr.fits', ],
            'B': ['f001a066.fits', ],
        }

        if not os.path.isdir(TEST_DATA_PATH):
            os.makedirs(TEST_DATA_PATH)

        config = fits2rgb.core.DEFAULT_CONFIG.copy()
        config['channels'] = channels
        config['options']['image-dir'] = TEST_DATA_PATH
        config['options']['out-name'] = 'm31_rgb_log.fits'
        config['options']['reproj-method'] = 'interp'
        config['options']['color-scale'] = 'log'

        with open(cfg_fname, 'w') as f:
            json.dump(config, f, indent=2)

        fits2rgb.core.main(['-c', cfg_fname])

    def test_rgb_lin(self, cfg_fname="test_lin_cfg.json"):
        print("***Testing m31, linear scale, no reprojection***")
        download_test_data()

        channels = {
            'R': ['f001a25o.fits', ],
            'G': ['f001a1hr.fits', ],
            'B': ['f001a066.fits', ],
        }

        if not os.path.isdir(TEST_DATA_PATH):
            os.makedirs(TEST_DATA_PATH)

        config = fits2rgb.core.DEFAULT_CONFIG.copy()
        config['channels'] = channels
        config['options']['image-dir'] = TEST_DATA_PATH
        config['options']['out-name'] = 'm31_rgb_lin.fits'
        config['options']['reproj-method'] = None
        config['options']['color-scale'] = 'lin'
        config['options']['out-shape'] = None

        with open(cfg_fname, 'w') as f:
            json.dump(config, f, indent=2)

        fits2rgb.core.main(['-c', cfg_fname])

    def test_rgb_res(self, cfg_fname="test_res_cfg.json"):
        print("***Testing m31, log scale, resample***")
        download_test_data()

        channels = {
            'R': ['f001a25o.fits', ],
            'G': ['f001a1hr.fits', ],
            'B': ['f001a066.fits', ],
        }

        if not os.path.isdir(TEST_DATA_PATH):
            os.makedirs(TEST_DATA_PATH)

        config = fits2rgb.core.DEFAULT_CONFIG.copy()
        config['channels'] = channels
        config['options']['image-dir'] = TEST_DATA_PATH
        config['options']['out-name'] = 'm31_rgb_res.fits'
        config['options']['reproj-method'] = 'interp'
        config['options']['color-scale'] = 'log'
        config['options']['out-shape'] = (1024, 1024)

        with open(cfg_fname, 'w') as f:
            json.dump(config, f, indent=2)

        fits2rgb.core.main(['-c', cfg_fname])

if __name__ == '__main__':
    mytest = TestFits2Rgb()
    mytest.test_rgb_res()
    mytest.test_rgb_lin()
    mytest.test_rgb_log()
