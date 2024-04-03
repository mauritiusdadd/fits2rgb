**********
User Guide
**********

Getting sample data
===================

In this guide we will use three FITS images of Andromeda galaxy (M31) that can be downloaded from https://esahubble.org/projects/fits_liberator/m31data.
After downloading and unpacking the zip files from the previous link, you should end with the following three FITS files ``f001a066.fits``, ``f001a1hr.fits``, and ``f001a25o.fits``, that contain pictures of M31 taken in the B, R, and I bands, respectively.
Finally, create a new directory named ``m31_images`` and move these files in it.

.. image:: pics/m31_mono_bands.png

Configuration file
==================

Fits2rgb needs a configuration file to know how and which files to combine. A default configuration file can be obtained using the command

.. code-block:: bash

    fits2rgb -d

this will create a file named ``fits2rgb.json`` that contains the following json data

.. code-block:: json

    {
        "channels": {
            "R": [],
            "G": [],
            "B": []
        },
        "options": {
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
            "out-shape": null,
            "out-dtype": "int16"
        }
    }

Channels section
----------------

The ``"channels"`` section describes how to combine the input FITS files.
It must have only three entries, one for each RGB channel.
The name of the entries will be the names of the HDUs in the output RGB FITS file.
The values of each entry is a list of FITS file that need to be combined.

For example ``"R": ["a.fits", "b.fits", "c.fits"]`` means that the three files ``a.fits``, ``b.fits``, and ``c.fits`` should be coadded into a new HDU named ``"R"``

.. note::
    The default names are ``"R"``, ``"G"``, and ``"B"`` but can be whatever you like.
    The first entry will always be the red channel, regardless of its name, an so for the second and the third ones, that will always be the green and blue channels respectively.

Options section
----------------

The ``"options"`` section is used to set the coaddition options.

* ``image-dir``: specify in which directory the input fits files are.
* ``out-name``: the name of the output file containing the RGB image, without the file extension.
* ``out-dir``: the directory where the output files are saved
* ``combine-function``: specify how to coadd the input fits files. Available functions are ``"mean"``, ``"sum"``, ``"std"``, ``"min"``, and ``"max"``.
* ``reproj-method``: specify how to reproject the input images. In can be either ``"interp"`` (fast interpolation), ``"exact"`` (slow but more accurate) or ``null`` (no reprojection). If no reprojection is used, images are assumed to be already registered and to have the same shape, an error is raised otherwise. Note also that if no reprojection is used, the output image will always have the same shape of the input ones, since no rebinning is done.
* ``nsamples``: maximum number of random pixels used to compute the statistics on each channel coadded image
* ``max-reject``: sigma for the sigma clipping
* ``krej``: number of iteration for the sigma clipping
* ``tile-size``: Images are processed in small tiles. This indicate the size of the tiles.
* ``contrast``: The contrast value, see
* ``gray-level``: The gray level, see
* ``color-scale``: The colo scale, can be either ``"log"`` or ``"linear"``. See
* ``out-shape``: The shape of the output image in the format ``[width, height]``. Can be also ``null``, in which case the output shape is computed automatically.
* ``out-dtype``: The data type of the output image, can be ``"int8"``, ``"int16"``, ``"int32"``, ``"int"``, ``"float16"``, ``"float32"``, ``"float"``. Since this program is thought to generate RGB images for visualization, the best data type is ``"int16"`` which produces relatively small files with only a minor loss of information.

Running fits2rgb
================

To run fits2rgb, just run it from the command line as

.. code-block:: bash

    fits2rgb

If the configuration file has a name different from the default one, you can use ``-c`` to specify it

.. code-block:: bash

    fits2rgb -c myconfig.json


For exampl, the following configuration to get a log-scale RGB image of approximately 1024x1024 pixels.

.. code-block:: json

    {
      "channels": {
        "R": ["f001a25o.fits"],
        "G": ["f001a1hr.fits"],
        "B": ["f001a066.fits"]
      },
      "options": {
        "image-dir": "/home/daddona/projects/python-fits2rgb/test/data",
        "out-name": "m31_rgb_res.fits",
        "out-dir": ".",
        "combine-function": "mean",
        "reproj-method": "interp",
        "nsamples": 1000,
        "max-reject": 5,
        "krej": 5,
        "tile-size": 256,
        "contrast": 5,
        "gray-level": 0.3,
        "color-scale": "log",
        "out-shape": [1024, 1024],
        "out-dtype": "int16"
      }
}

If we save this to the file ``m31_rgb_log.json``, then we can use the command

.. code-block:: bash

    fits2rgb -c m31_rgb_log.json

to generate the following RGB FITS image.

.. image:: pics/ds9_m31_rgb.png