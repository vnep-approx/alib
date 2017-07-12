__author__ = 'Tom Koch (tkoch@inet.tu-berlin.de)'

from alib import datamodel, mip


class TestModelCreator:
    def setup(self):
        self.substrate = datamodel.Substrate("sub1")
        self.request = datamodel.LinearRequest("req1")
        self.scenario = datamodel.Scenario("Sen1", self.substrate,
                                           [self.request])

    def test_no_edge_mapping_model(self):
        # SUBSTRATE: - NODES
        self.substrate.add_node('u', ["FW", "DPI"], capacity={"FW": 2, "DPI":
            2}, cost={"FW": 2, "DPI":
            2})
        self.substrate.add_node('v', ["FW"], capacity={"FW": 2}, cost={"FW": 2})
        #           - EDGES
        self.substrate.add_edge('u', 'v', capacity=2)
        # REQUEST NODES AND EDGES
        self.request.profit = 1
        self.request.add_node('i', 2, "FW")
        self.request.add_node('j', 2, "DPI")
        self.request.add_edge('i', 'j', 2)
        # REQUEST LATENCY
        mc = mip.ClassicMCFModel(self.scenario)
        mc.init_model_creator()

    def test_init_model_cheap(self):
        # SUBSTRATE: - NODES
        self.substrate.add_node('v1', ["DPI"], capacity={"FW": 2, "DPI":
            2}, cost={"FW": 2, "DPI":
            2})
        self.substrate.add_node('v2', ["FW"], capacity={"FW": 2}, cost={"FW": 2})
        self.substrate.add_node('v3', ["FW"], capacity={"FW": 2, "DPI":
            2}, cost={"FW": 2, "DPI":
            2})
        #           - EDGES
        self.substrate.add_edge('v1', 'v2', capacity=2)
        self.substrate.add_edge('v2', 'v3', capacity=2)
        self.substrate.add_edge('v3', 'v1', capacity=2)
        # REQUEST NODES AND EDGES
        self.request.profit = 5
        self.request.add_node('i1', 2, "FW", allowed_nodes=['v2'])
        self.request.add_node('i2', 2, "DPI")
        self.request.add_node('i3', 2, "FW")
        self.request.add_edge('i1', 'i2', 2)
        self.request.add_edge('i2', 'i3', 2)
        mc = mip.ClassicMCFModel(self.scenario)
        mc.init_model_creator()

    def test_edge_mapping_on_multiple_edges_model(self):
        # SUBSTRATE: - NODES
        self.substrate.add_node('u', ["DPI"], capacity={"DPI":
                                                            2}, cost={"DPI":
                                                                          2})
        self.substrate.add_node('v', ["FW"], capacity={"FW": 2}, cost={"FW": 2})
        self.substrate.add_node('w', ["X"], capacity={"X": 2}, cost={"X": 2})
        self.substrate.add_edge('u', 'v', capacity=9)
        self.substrate.add_edge('v', 'w', capacity=9)

        self.request.profit = 1
        self.request.add_node('i', 2, "DPI")
        self.request.add_node('j', 2, "X")
        # REQUEST LATENCY
        self.request.add_edge('i', 'j', 1)
        mc = mip.ClassicMCFModel(self.scenario)
        mc.init_model_creator()

        # mc.compute_integral_solution()
