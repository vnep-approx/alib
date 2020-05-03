__author__ = 'Tom Koch (tkoch@inet.tu-berlin.de)'
import copy

import pytest

from alib import datamodel


class TestLinearRequest:
    def setup(self):
        self.request = datamodel.LinearRequest("req1")

    def initialize_request(self):
        self.request.add_node("i", 1, "t1")
        self.request.add_node("j", 2, "t2")
        self.request.add_node("k", 3, "t1")

        self.request.add_edge("i", "j", 1.0)
        self.request.add_edge("j", "k", 1.0)

    def test_addNode(self):
        assert self.request.get_nodes() == set()
        assert self.request.sequence == []
        self.initialize_request()
        assert self.request.get_nodes() == {"i", "j", "k"}
        assert self.request.sequence == ["i", "j", "k"]

    def test_add_edge(self):
        assert [] == self.request.sequence
        assert self.request.edges == set()

        self.initialize_request()

        assert self.request.edges == {("i", "j"), ("j", "k")}

    def test_out_edges(self):
        self.initialize_request()
        assert self.request.get_out_edge("i") == ("i", "j")
        assert self.request.get_out_edge("k") == None

        # Modify out edges to get
        self.request.out_edges["i"].append(("i", "k"))
        with pytest.raises(datamodel.LinearRequestError):
            self.request.get_out_edge("i")

        with pytest.raises(datamodel.LinearRequestError):
            self.request.add_edge("i", "k", 1.0)


class TestRequest:
    def setup(self):
        self.request = datamodel.Request("req1")

    def initialize_request(self):
        self.request.add_node("i", 1, "t1")
        self.request.add_node("j", 2, "t2")
        self.request.add_node("k", 3, "t1")

        self.request.add_edge("i", "j", 1.0)
        self.request.add_edge("j", "k", 1.0)

    def test_add_node(self):
        assert self.request.get_nodes() == set()

        self.initialize_request()

        assert self.request.get_nodes() == {"i", "j", "k"}

    def test_add_edge(self):
        assert self.request.edges == set()

        self.initialize_request()

        assert self.request.edges == {("i", "j"), ("j", "k")}



class TestScenario:
    def setup(self):
        self.substrate = datamodel.Substrate("sub1")
        self.request = datamodel.Request("req1")
        self.scenario = datamodel.Scenario("Sen1", self.substrate, [self.request])

    def test_validate_types(self):
        # REQUEST NODES AND EDGES
        self.request.add_node('i1', 2, "FW")
        self.request.add_node('i2', 2, "DPI")
        self.request.add_node('i3', 2, "FW")
        self.request.add_edge('i1', 'i2', 2)
        self.request.add_edge('i2', 'i3', 2)
        # REQUEST TYPES
        print "Request requires types"
        print self.request.get_required_types()
        # SUBSTRATE: - NODES
        self.substrate.add_node('v1', ["FW", "DPI"], {"FW": 1, "DPI": 2}, {"FW": 1, "DPI": 1})
        self.substrate.add_node('v2', ["FW"], {"FW": 1}, {"FW": 1})
        self.substrate.add_node('v3', ["FW", "DPI"], {"FW": 1, "DPI": 2}, {"FW": 1, "DPI": 1})
        self.substrate.add_node('v4', ["FW"], {"FW": 1}, {"FW": 1})
        self.substrate.add_node('v5', ["FW", "DPI"], {"FW": 1, "DPI": 2}, {"FW": 1, "DPI": 1})
        #           - EDGES
        self.substrate.add_edge('v1', 'v2')
        self.substrate.add_edge('v2', 'v3')
        self.substrate.add_edge('v3', 'v4')
        self.substrate.add_edge('v4', 'v5')
        print "Substrate offers Types", self.substrate.get_types()
        # SCENARIO
        assert self.scenario.validate_types()


# deprecated
class TestSubstrate:
    def setup(self):
        self.substrate = datamodel.Substrate("substrate1")

    def test_addNode(self):
        assert not ("node1" in self.substrate.get_nodes())
        self.substrate.add_node('node1', ["FW"], {"FW": 1}, {"FW": 1})
        assert "node1" in self.substrate.get_nodes()

    def test_addEdge(self):
        self.substrate.add_node('node1', ["FW"], {"FW": 1}, {"FW": 1})
        self.substrate.add_node('node2', ["DPI"], {"DPI": 1}, {"DPI": 1})
        self.substrate.add_edge('node1', 'node2', 1)
        assert self.substrate.get_number_of_edges() == 2

    def test_edgeProperties(self):
        self.substrate.add_node('node1', ["FW"], {"FW": 1}, {"FW": 1})
        self.substrate.add_node('node2', ["DPI"], {"DPI": 1}, {"DPI": 1})
        self.substrate.add_edge('node1', 'node2', 2, 2, 2)
        assert self.substrate.get_edge_cost(('node1', 'node2')) == 2

    def test_get_out_edge(self):
        self.substrate.add_node('u', ["FW"], {"FW": 1}, {"FW": 1})
        self.substrate.add_node('v', ["FW"], {"FW": 1}, {"FW": 1})
        self.substrate.add_node('w', ["DPI"], {"DPI": 1}, {"DPI": 1})
        self.substrate.add_edge("u", "v")

        assert self.substrate.get_nodes_by_type("DPI") == ["w"]

    def test_error_handling(self):
        # Calling add_node without a list of supported types:
        with pytest.raises(datamodel.SubstrateError):
            copy.deepcopy(self.substrate).add_node("u", "types", {"types": 1.0}, {"types": 1.0})
        # Calling add_node with incomplete capacity dict:
        with pytest.raises(datamodel.SubstrateError):
            copy.deepcopy(self.substrate).add_node("u", ["t1", "t2"], {"t1": 1.0}, {"t1": 1.0, "t2": 1.0})
        # Calling add_edge with non-existing nodes:
        with pytest.raises(datamodel.SubstrateError):
            copy.deepcopy(self.substrate).add_edge("u", "v")

    def test_node_and_edge_getters(self):
        self.substrate.add_node('u', ["t1", "t2"], {"t1": 1, "t2": 0.5}, {"t1": 2, "t2": 3})
        self.substrate.add_node('v', ["t1"], {"t1": 2}, {"t1": 2})
        self.substrate.add_node('w', ["t1"], {"t1": 1.5}, {"t1": 2})

        assert self.substrate.get_node_capacity("v")  == pytest.approx(2.0)
        assert self.substrate.get_node_type_capacity("u", "t1") == 1
        assert self.substrate.average_node_capacity("t1") == pytest.approx(1.5)
        assert self.substrate.get_total_node_resources("t1") == pytest.approx(4.5)
        assert set(self.substrate.get_nodes_by_type("t1")) == {"u", "v", "w"}

        # now, some edges:
        self.substrate.add_edge("u", "v", capacity=1.2, cost=1.3, bidirected=True)
        self.substrate.add_edge("v", "w",  capacity=1.2, cost=1.3, bidirected=True)
        assert self.substrate.get_total_edge_resources() == pytest.approx(4.8)  # doubled because edge is bidirected
        assert self.substrate.get_edge_capacity(("u", "v")) == pytest.approx(1.2)

        # add edges with a different capacity:
        self.substrate.add_edge("u", "w", capacity=4.8, cost=1.3, bidirected=True)
        assert self.substrate.average_edge_capacity() == pytest.approx(2.4)


class TestSubstrateX:
    def setup(self):
        self.substrate = datamodel.Substrate("paper_example_substrate")
        self.substrate.add_node("u", ["t1"], {"t1": 1}, {"t1": 0.0})
        self.substrate.add_node("v", ["t1"], {"t1": 2}, {"t1": 0.0})
        self.substrate.add_node("w", ["t1", "t2"], {"t1": 3, "t2": 5}, {"t1": 0.0, "t2": 0.0})
        self.substrate.add_node("x", ["t2"], {"t2": 1}, {"t2": 0.0})
        self.substrate.add_edge("u", "v", capacity=1000, bidirected=True)
        self.substrate.add_edge("u", "w", capacity=1, bidirected=False)
        self.substrate.add_edge("u", "x", capacity=100, bidirected=False)
        self.substrate_x = datamodel.SubstrateX(self.substrate)

    def test_default_substrate_attributes_work(self):
        assert self.substrate_x.nodes
        assert self.substrate_x.node
        assert self.substrate_x.node
        assert self.substrate_x.edges
        assert self.substrate_x.substrate_node_resources
        assert self.substrate_x.substrate_edge_resources
        assert self.substrate_x.substrate_resources
        assert self.substrate_x.substrate_resources
        assert self.substrate_x.substrate_resource_capacities

    def test_get_valid_edges(self):
        assert self.substrate_x.get_valid_edges(-100) == {('u', 'v'), ('v', 'u'), ('u', 'w'), ('u', 'x')}
        assert self.substrate_x.get_valid_edges(0) == {('u', 'v'), ('v', 'u'), ('u', 'w'), ('u', 'x')}
        assert self.substrate_x.get_valid_edges(1) == {('u', 'v'), ('v', 'u'), ('u', 'w'), ('u', 'x')}
        assert self.substrate_x.get_valid_edges(50) == {('u', 'v'), ('v', 'u'), ('u', 'x')}
        assert self.substrate_x.get_valid_edges(100) == {('u', 'v'), ('v', 'u'), ('u', 'x')}
        assert self.substrate_x.get_valid_edges(150) == {('u', 'v'), ('v', 'u')}
        assert self.substrate_x.get_valid_edges(1000) == {('u', 'v'), ('v', 'u')}
        assert self.substrate_x.get_valid_edges(1001) == set()

    def test_get_valid_nodes(self):
        assert self.substrate_x.get_valid_nodes("t1", -1) == {"u", "v", "w"}
        assert self.substrate_x.get_valid_nodes("t1", 0) == {"u", "v", "w"}
        assert self.substrate_x.get_valid_nodes("t1", 1) == {"u", "v", "w"}
        assert self.substrate_x.get_valid_nodes("t1", 1.5) == {"v", "w"}
        assert self.substrate_x.get_valid_nodes("t1", 2) == {"v", "w"}
        assert self.substrate_x.get_valid_nodes("t1", 2.5) == {"w"}
        assert self.substrate_x.get_valid_nodes("t1", 3) == {"w"}
        assert self.substrate_x.get_valid_nodes("t1", 5) == set()

        assert self.substrate_x.get_valid_nodes("t2", -1) == {"w", "x"}
        assert self.substrate_x.get_valid_nodes("t2", 1) == {"w", "x"}
        assert self.substrate_x.get_valid_nodes("t2", 3) == {"w"}
        assert self.substrate_x.get_valid_nodes("t2", 5) == {"w"}
        assert self.substrate_x.get_valid_nodes("t2", 6) == set()
