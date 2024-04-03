.. |github_mark| image:: pics/github-mark.png
   :height: 1em
   :target: github_repo

Installing a Packaged Release
=============================

The simplest way to install fits2rgb is using ``pip`` and, since it is a good practice to not mess up the system-wide python environment, you should install this program in a virtual environment. If you don't have a virtual environment yet, you can create one with the command

.. code-block:: bash

    python -m venv env_name

For example, to create a virtual environment called "astro", you can use the command

.. code-block:: bash

    python -m venv astro

and you can activate it with

.. code-block:: bash

    source astro/bin/activate

Then run

.. code-block:: bash

    pip install fits2rgb

    
After the installation, to update redmost to the most recent release, use

.. code-block:: bash

    pip install fits2rgb --upgrade

    
Installing from GitHub
======================

If you like to use the bleeding-edge version, you can install fits2rgb directly from the |github_mark| `GitHub repository <https://github.com/mauritiusdadd/fits2rgb>`_

.. code-block:: bash

    git clone 'https://github.com/mauritiusdadd/fits2rgb.git'
    cd fits2rgb
    pip install .

After the installation, to update to the most recent commit use

.. code-block:: bash
    git pull
    pip install . --upgrade
