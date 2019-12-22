
# Overview

The **alib** (short for **a library**) provides a common **Python 2.7** basis for our **Virtual Network Embedding Problem (VNEP)** approximation framework. As such, it contains (among other things) 
- A common **[data model](alib/datamodel.py)** to capture the notions of **substrate graphs** (physical networks), **request graphs** (virtual networks), and **embeddings** of requests to a substrate, **scenarios**, i.e. bundling multiple requests to be embedded on a common substrate.
- A common **[scenario generation](alib/scenariogeneration.py)** framework to generate cartesian products of parameter spaces and to generate scenarios accordingly (at random).
- A common **[scenario execution](alib/run_experiment.py)** framework to execute experments in parallel and using arbitrarily many different parameter configurations.
- A common base for solving **[integer/linear programs](alib/modelcreator.py)** pertaining to the VNEP together with an implementation of the **[classic multi-commodity flow formulation](alib/mip.py)** for the VNEP.

# Dependencies and Requirements

The alib library requires Python 2.7. Required python libraries: gurobipy, numpy, cPickle, networkx 1.9, matplotlib. 

Note: Unfortunately, newer versions of networkx cannot parse many of the topology zoo's \*.gml files. This problem can be avoided by downgrading to networkx 1.9 ("(sudo) pip install networkx==1.9"), or by removing certain node annotations and any duplicate edges from the \*.gml files.

Gurobi must be installed and the .../gurobi64/lib directory added to the environment variable LD_LIBRARY_PATH.

For generating and executing (etc.) experiments, the environment variable ALIB_EXPERIMENT_HOME must be set to a path,
such that the subfolders input/ output/ and log/ exist.

**Note**: Our source was only tested on Linux (specifically Ubuntu 14/16).  

# Installation

To install **alib**, we provide a setup script. Simply execute from within alib's root directory: 

```
pip install .
```

Furthermore, if the code base will be edited by you, we propose to install it as editable:
```
pip install -e .
```
When choosing this option, sources are not copied during the installation but the local sources are used: changes to
the sources are directly reflected in the installed package.

We generally propose to install **alib** into a virtual environment.

# Usage

You may either use our code via our API by importing the library or via our command line interface:

```
python -m alib.cli

Usage: cli.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  generate_scenarios
  inspect_cactus_request_graph_generation
  pretty_print
  start_experiment
```

# Tests

The test directory contains a large number of tests to check the correctness of our implementation and might also be useful
to understand the code. 

To execute the tests, simply execute pytest in the test directory.

```
pytest .
```

# API Documentation

We provide a basic template to create an API documentatio using **[Sphinx](http://www.sphinx-doc.org)**. 

To create the documentation, simply execute the makefile in **docs/**. Specifically, run for example

```
make html
```

to create the HTML documentation.

Note that **alib** must lie on the PYTHONPATH. If you use a virtual environment, we propose to install sphinx within the
virtual environment (using **pip install spinx**) and executing the above from within the virtual environment. 

# Contact

If you have any questions, simply write a mail to mrost(AT)inet.tu-berlin(DOT)de.

# Acknowledgement

Major parts of this code were developed under the support of the **German BMBF Software
Campus grant 01IS1205**.