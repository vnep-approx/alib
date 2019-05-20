#!/bin/bash

python -m alib.cli convert_topology_zoo_gml_to_yml topologyZoo_orig/ topologyZoo/ --consider_disconnected > conversion.log
