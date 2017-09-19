__author__ = 'Tom Koch (tkoch@inet.tu-berlin.de)'

from alib import datamodel, solutions


class TestScenarioSolution:
    def setup(self):
        self.substrate = datamodel.Substrate("sub1")
        self.request = datamodel.Request("req1")
        self.scenario = datamodel.Scenario("Sen1", self.substrate, [self.request])
        self.mapping = solutions.Mapping("map1", self.request, self.substrate,
                                         True)
        self.scenariosolution = solutions.IntegralScenarioSolution("Solution1", self.scenario)

    def test_validate_solution(self):
        # REQUEST NODES AND EDGES
        self.request.add_node('i1', 2, "FW")
        self.request.add_node('i2', 2, "DPI")
        self.request.add_node('i3', 2, "FW")
        self.request.add_edge('i1', 'i2', 2)
        self.request.add_edge('i2', 'i3', 2)
        # SUBSTRATE: - NODES
        self.substrate.add_node('v1', ["FW", "DPI"], {"FW": 2, "DPI": 2}, {"FW": 1, "DPI": 1})
        self.substrate.add_node('v2', ["FW"], {"FW": 2}, {"FW": 1})
        self.substrate.add_node('v3', ["FW", "DPI"], {"FW": 2, "DPI": 2}, {"FW": 1, "DPI": 1})
        self.substrate.add_node('v4', ["FW"], {"FW": 2}, {"FW": 1})
        self.substrate.add_node('v5', ["FW", "DPI"], {"FW": 2, "DPI": 2}, {"FW": 1, "DPI": 1})
        #           - EDGES
        self.substrate.add_edge('v1', 'v2', capacity=2.0)
        self.substrate.add_edge('v2', 'v3', capacity=2.0)
        self.substrate.add_edge('v3', 'v4', capacity=2.0)
        self.substrate.add_edge('v4', 'v5', capacity=2.0)
        # MAPPING 
        self.mapping.map_node('i1', 'v1')
        self.mapping.map_node('i2', 'v3')
        self.mapping.map_node('i3', 'v5')
        self.mapping.map_edge(('i1', 'i2'), (('v1', 'v2'), ('v2', 'v3')))
        self.mapping.map_edge(('i2', 'i3'), (('v3', 'v4'), ('v4', 'v5')))
        self.scenariosolution.add_mapping(self.request, self.mapping)
        assert self.scenariosolution.validate_solution()
