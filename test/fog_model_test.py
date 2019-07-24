# MIT License
#
# Copyright (c) 2019 Balazs Nemeth
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


from alib.scenariogeneration_for_fog_model import CactusGraphGenerator, SyntheticSeriesParallelDecomposableRequestGenerator

import networkx as nx
import random


def find_all_cycles(G, source=None, cycle_length_limit=None):
    """forked from networkx dfs_edges function. Assumes nodes are integers, or at least
    types which work with min() and > .

    Taken from: https://gist.github.com/joe-jordan/6548029, full credits to him.
    """
    if source is None:
        # produce edges for all components
        nodes = [i.pop() for i in nx.connected_components(G)]
    else:
        # produce edges for components with source
        nodes = [source]
    # extra variables for cycle detection:
    cycle_stack = []
    output_cycles = set()

    def get_hashable_cycle(cycle):
        """cycle as a tuple in a deterministic order."""
        m = min(cycle)
        mi = cycle.index(m)
        mi_plus_1 = mi + 1 if mi < len(cycle) - 1 else 0
        if cycle[mi - 1] > cycle[mi_plus_1]:
            result = cycle[mi:] + cycle[:mi]
        else:
            result = list(reversed(cycle[:mi_plus_1])) + list(reversed(cycle[mi_plus_1:]))
        return tuple(result)

    for start in nodes:
        if start in cycle_stack:
            continue
        cycle_stack.append(start)

        stack = [(start, iter(G[start]))]
        while stack:
            parent, children = stack[-1]
            try:
                child = next(children)

                if child not in cycle_stack:
                    cycle_stack.append(child)
                    stack.append((child, iter(G[child])))
                else:
                    i = cycle_stack.index(child)
                    if i < len(cycle_stack) - 2:
                        output_cycles.add(get_hashable_cycle(cycle_stack[i:]))

            except StopIteration:
                stack.pop()
                cycle_stack.pop()

    return [list(i) for i in output_cycles]


def test_cactus_generation():
    for n in xrange(5, 30):
        for ctr in xrange(2, 9):
            ctr = ctr / 10.0
            for ccr in xrange(2, 9):
                ccr = ccr / 10.0
                for tcr in xrange(2, 9):
                    tcr = tcr / 10.0
                    for seed in xrange(1, 5):
                        r = random.Random(seed)
                        G = CactusGraphGenerator(n=n, cycle_tree_ratio=ctr, cycle_count_ratio=ccr,
                                                 tree_count_ratio=tcr, random=r).generate_cactus()
                        G_nx = nx.Graph()
                        G_nx.add_edges_from(G.edges)
                        cycle_edges = set()
                        for cycle in find_all_cycles(G_nx):
                            if len(cycle) == 2:
                                raise Exception("2 length cycle found")
                            for i,j in zip(cycle[:-1], cycle[1:]):
                                if (i,j) in cycle_edges or (j,i) in cycle_edges:
                                    raise Exception("The edge {}, {} violates the cactus property, because it is containted in 2 cycles, "
                                                    "one of them is {}".format(i,j,cycle))
                                else:
                                    cycle_edges.add((i,j))
                        print "OK: n:{}, ctr: {}, ccr: {}, tcr: {}, seed: {}\n".format(n, ctr, ccr, tcr, seed)


def test_spd_generation():
    for n in xrange(2, 20):
            # different r might not terminate... But this is how range splitter is implemented...
            r = 0.5
            for sp in xrange(1, 9):
                sp = sp / 10.0
                sspdrg = SyntheticSeriesParallelDecomposableRequestGenerator()
                sspdrg.range_splitter = r
                sspdrg.parallel_serial_ratio = sp
                G = sspdrg.series_parallel_decomposable_generator(n)
                print "OK: n: {}, r: {}, sp: {}, edge_count: {}, node_count: {}\n".\
                                format(n, r, sp, G.number_of_edges(), G.number_of_nodes())
                # NOTE: connected_components is not implemented for directed type
                # cycles = find_all_cycles(G)
                # if len(cycles) > 0:
                #     raise Exception("SPD cannot have cycles: {}".format(cycles))


# TODO: add it to pytest, like the rest of the framework
if __name__ == "__main__":
    test_spd_generation()
    test_cactus_generation()

