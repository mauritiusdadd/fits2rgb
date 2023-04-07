#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPEX - SPectra EXtractor.

Extract spectra from spectral data cubes.

Copyright (C) 2022  Maurizio D'Addona <mauritiusdadd@gmail.com>
"""
import os
import json

import unittest

import numpy as np
from astropy.io import fits

import fits2rgb


class TestFits2Rgb(unittest.TestCase):

    def test_fits_2_rgb(self, inp_img_dir='test_images',
                        cfg_fname="test_config.json"):
        channel_names = ['R', 'G', 'B']
        channels = {}

        if not os.path.isdir(inp_img_dir):
            os.makedirs(inp_img_dir)

        for k, channel_name in enumerate(channel_names):
            channels[channel_name] = []
            for i in range(k, 3):
                image_index = len(channel_names) - i
                ch_image_fname = f"test_{channel_name}_{image_index:d}"
                image_data = np.random.random(size=(50, 50))
                hdu = fits.PrimaryHDU(image_data)
                hdu.writeto(
                    os.path.join(
                        inp_img_dir,
                        ch_image_fname
                    ),
                    overwrite=True
                )
                channels[channel_name].append(ch_image_fname)

        config = fits2rgb.DEFAULT_CONFIG.copy()
        config["channels"] = channels
        config['options']['image-dir'] = inp_img_dir

        with open(cfg_fname, 'w') as f:
            json.dump(config, f, indent=2)

        fits2rgb.main(['-c', cfg_fname])


if __name__ == '__main__':
    mytest = TestFits2Rgb()
    mytest.test_fits_2_rgb()
