# MIT License
#
# Copyright (c) 2016-2018 Matthias Rost, Elias Doehne, Tom Koch, Alexander Elvers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import jsonpickle
import os

'''
    This functionality is not used anymore, I believe.
'''

class SuitableSubstrates(object):

    def __init__(self):
        self.names = []
        self.nodes_to_names = {}
        self.names_to_nodes = {}
        self.substrates = {}

    def add_entry(self, name, number_of_nodes, substrate):
        self.names.append(name)
        if number_of_nodes in self.nodes_to_names:
            self.nodes_to_names[number_of_nodes].append(name)
        else:
            self.nodes_to_names[number_of_nodes] = [name]
        self.names_to_nodes[name] = number_of_nodes
        self.substrates[name] = substrate


    def get_names(self):
        return self.names

    def get_substrate(self, name):
        return self.substrates[name]

    def print_it(self):
        print "\nPrinting suitable substrates...\n"
        print self.names
        print self.nodes_to_names
        print self.names_to_nodes
        print self.substrates
        print "\n\n"

    def prune_topologies(self):
        to_delete = []
        to_keep = []
        for number_of_nodes in self.nodes_to_names.keys():
            to_keep.append(self.nodes_to_names[number_of_nodes][0])
            to_delete.extend(self.nodes_to_names[number_of_nodes][1:])

        self.names = to_keep
        for topology_to_delete in to_delete:
            self.nodes_to_names[self.names_to_nodes[topology_to_delete]].remove(topology_to_delete)
            del self.names_to_nodes[topology_to_delete]
            del self.substrates[topology_to_delete]

#changed here for server subserver architecture - added ../../       !!!!!
def unpickle_suitable_substrates(path= "input/suitable_substrates.json", prefix=""):
    with open(os.getcwd() + "/" + prefix + path) as f:
        suitable_substrates = jsonpickle.decode(f.read(), keys=True)
    return suitable_substrates

def pickle_suitable_substrates(suitable_substrates, path= "input/suitable_substrates.json", prefix=""):
    path_to_file = os.getcwd() + "/" + prefix + path
    print path_to_file
    with open(path_to_file, "w") as f:
        f.write(jsonpickle.encode(suitable_substrates, keys=True))

def unpickle_pruned_suitable_substrates(path="input/pruned_suitable_substrates.json"):
    with open(os.getcwd() + "/" + path) as f:
        suitable_substrates = jsonpickle.decode(f.read(), keys=True)
    return suitable_substrates

def pickle_pruned_suitable_substrates(suitable_substrates, path="input/pruned_suitable_substrates.json"):
    path_to_file = os.getcwd() + "/" + path
    print path_to_file
    with open(path_to_file, "w") as f:
        f.write(jsonpickle.encode(suitable_substrates, keys=True))

