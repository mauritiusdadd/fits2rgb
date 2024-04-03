[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7808276.svg)](https://doi.org/10.5281/zenodo.7808276)
# fits2rgb 

A simple program for making coadded RGB FITS images for visualization. 

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