[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7808276.svg)](https://doi.org/10.5281/zenodo.7808276) 
[![Testing](https://github.com/mauritiusdadd/fits2rgb/actions/workflows/test_linux.yml/badge.svg)](https://github.com/mauritiusdadd/fits2rgb/actions/workflows/test_linux.yml)
[![Documentation Status](https://readthedocs.org/projects/fits2rgb/badge/?version=latest)](https://fits2rgb.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/mauritiusdadd/fits2rgb/badge.svg)](https://coveralls.io/github/mauritiusdadd/fits2rgb)

# fits2rgb 

A simple program for making coadded RGB FITS images for visualization. 

Full documentation is available at https://fits2rgb.readthedocs.io/en/latest/index.html

# Installation

### Installing a Packaged Release

The simplest way to install fits2rgb is using ``pip`` and, since it is a good practice to not mess up the system-wide python environment, you should install this program in a virtual environment. If you don't have a virtual environment yet, you can create one with the command

```
python -m venv env_name
```

For example, to create a virtual environment called "astro", you can use the command

```
python -m venv astro
```

and you can activate it with

```
source astro/bin/activate
```
Then run

```
pip install fits2rgb
```
    
After the installation, to update redmost to the most recent release, use

```
pip install fits2rgb --upgrade
```
    
### Installing from GitHub

If you like to use the bleeding-edge version from this repository, do

```
git clone 'https://github.com/mauritiusdadd/fits2rgb.git'
cd fits2rgb
pip install .
```

After the installation, to upgrade to most recent commit use

```
git pull
pip install . --upgrade
```

Then run ```pytest``` to check everything is ok.

# Usage

To create a default configuration file, use the command

```
fits2rgb -d
```

this will create a file named fits2rgb.json that you can edit to specify how and which files you want to coadd.
Then you can run the program by simply invoking it from the command line. 

```
fits2rgb
```

You can also specify which configuration file to use, for example

```
fits2rgb -c fits2rgb.json
```

For more dtails, please refer to the [documentation](https://fits2rgb.readthedocs.io/en/latest/index.html).