.. _installation:

Installation
============

Prerequisites
-------------

- Python 3.11 or higher
- Linux or macOS

From source (recommended)
-------------------------

First, clone the ProtFlow repository. Use either one of the following commands:

.. code-block:: bash

    git clone https://github.com/mabr3112/ProtFlow.git
    git clone https://github.com/mabr3112/ProtFlow.git

ProtFlow requires python >= 3.11. You can either install it into an existing environment, or you can create a new one with like this:

.. code-block:: bash

    conda create -n protflow python=3.11
    conda activate protflow

Now install the package. We recommend installing in development mode, because this allows you to easily update ProtFlow in the future.

.. code-block:: bash

    cd ProtFlow
    pip install -e .

After the install, instantiate your config.py file. This can be done with the command below, which will copy the config.py file from protflow/config_template.py

.. code-block:: bash

    protflow-init-config

Finally, you want to verify that the installation was successful. You can do this by running the following command:

.. code-block:: bash

    python -c "import protflow; print(protflow.__version__); sys.exit(0)"

Concratulations! Now you can start using ProtFlow. Continue with setting up your configuration file in the :ref:`configuration` section.
