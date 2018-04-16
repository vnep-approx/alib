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

"""
Some convenience functions to generate simple, artificial substrate and request
graphs and scenarios for testing purposes.
"""

from . import datamodel, scenariogeneration


def get_test_request(number_of_nodes, name="test_request", demand=1.0):
    """
    Generate a complete graph as a single request.

    :param number_of_nodes:
    :param name:
    :param demand:
    :return:
    """
    test_request = datamodel.Request(name)

    for i in range(1, number_of_nodes + 1):
        test_request.add_node("{}_node_{}".format(name, i),
                              demand,
                              scenariogeneration.UNIVERSAL_NODE_TYPE)
    for i in range(1, number_of_nodes + 1):
        for j in range(i + 1, number_of_nodes + 1):
            test_request.add_edge("{}_node_{}".format(name, i),
                                  "{}_node_{}".format(name, j),
                                  demand)
    return test_request


def get_test_linear_request(number_of_nodes, name="test_request", demand=1.0):
    """
    Generate a simple chain as a LinearRequest object.

    :param number_of_nodes:
    :param name:
    :param demand:
    :return:
    """
    test_request = datamodel.LinearRequest(name)
    last_node = None
    for i in range(1, number_of_nodes + 1):
        new_node = "{}_node_{}".format(name, i)
        test_request.add_node(new_node,
                              demand,
                              scenariogeneration.UNIVERSAL_NODE_TYPE)
        if last_node is not None:
            test_request.add_edge(last_node, new_node, demand)
        last_node = new_node
    return test_request


def get_test_substrate(number_of_nodes, node_types=None, capacity=10.0):
    """
    Generate a complete graph as a substrate.

    :param number_of_nodes:
    :param name:
    :param demand:
    :return:
    """
    if node_types is None:
        node_types = [scenariogeneration.UNIVERSAL_NODE_TYPE]
    test_substrate = datamodel.Substrate("test_substrate")
    capacity = {nt: capacity for nt in node_types}
    cost = {nt: 1.0 for nt in node_types}
    for i in range(1, number_of_nodes + 1):
        test_substrate.add_node("test_substrate_node_{}".format(i), node_types, capacity, cost)
    for i in range(1, number_of_nodes + 1):
        for j in range(i + 1, number_of_nodes + 1):
            test_substrate.add_edge("test_substrate_node_{}".format(i),
                                    "test_substrate_node_{}".format(j),
                                    capacity['universal'], bidirected=True)
    test_substrate.initialize_shortest_paths_costs()
    return test_substrate


def get_test_scenario(number_of_requests=1,
                      request_size=2,
                      substrate_size=3,
                      request_demand=1.0,
                      substrate_capacity=10.0):
    sub = get_test_substrate(substrate_size, capacity=substrate_capacity)
    requests = []
    for i in range(1, 1 + number_of_requests):
        name = "test_req_{}".format(i)
        req = get_test_request(request_size,
                               name=name,
                               demand=request_demand)
        requests.append(req)
    return datamodel.Scenario("test_scenario", sub, requests)


def get_example_scenario_from_paper():
    sub = datamodel.Substrate("sub1")
    request = datamodel.LinearRequest("req1")
    example_scenario = datamodel.Scenario("Sen1", sub, [request])

    sub.add_node('v1', ["FW", "DPI"], {"FW": 2, "DPI": 1}, {"FW": 1, "DPI": 1})
    sub.add_node('v2', ["FW"], {"FW": 2}, {"FW": 1})
    sub.add_node('v3', ["FW", "DPI"], {"FW": 2, "DPI": 1}, {"FW": 1, "DPI": 1})
    #           - EDGES
    sub.add_edge('v1', 'v2', capacity=2)
    sub.add_edge('v2', 'v3', capacity=2)
    sub.add_edge('v3', 'v1', capacity=2)
    # REQUEST NODES AND EDGES
    request.add_node('i1', 1, "FW", allowed_nodes=["v1", "v2", "v3"])
    request.add_node('i2', 1, "DPI", allowed_nodes=["v3", "v1"])
    request.add_node('i3', 1, "FW", allowed_nodes=["v1"])
    request.add_edge('i1', 'i2', 2)
    request.add_edge('i2', 'i3', 2)
    request.profit = 10000
    return example_scenario
