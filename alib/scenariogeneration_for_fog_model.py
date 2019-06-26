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

from . import scenariogeneration as sg, datamodel


class ABBUseCaseRequestGenerator(sg.AbstractRequestGenerator):
    """
    Generates a industrial fog application described in
    Suter, Eidenbenz, Pignolet, Singla -- Fog Application Allocation for Automation Systems
    """

    EXPECTED_PARAMETERS = [
        'sensor_actuator_loop_count' # If not even one lower number is used
    ]

    def __init__(self, logger=None):
        super(ABBUseCaseRequestGenerator).__init__(logger=logger)
        # All parameters of the request generator are inicialized here and the same names are expected in 'raw_parameters'
        self.sensor_actuator_loop_count = None
        self.universal_node_type = 'universal'

    def _read_raw_parameters(self, raw_parameters):
        """
        Reads all expected parameters

        :param raw_parameters:
        :return:
        """
        try:
            self.sensor_actuator_loop_count = int(raw_parameters['sensor_actuator_loop_count'])
            if self.sensor_actuator_loop_count % 2 == 1:
                self.sensor_actuator_loop_count -= 1
        except KeyError as e:
            raise sg.ExperimentSpecificationError("Parameter not found in request specification: {keyerror}".format(keyerror=e))

    def _add_single_preprocessing_block(self, req, index, substrate_node):
        """
        Adds a single A-B-C-D-E block of preprocessor network to req with index. Returns the nodes where the rest of the application
        should be connected

        :param substrate_node:
        :param req:
        :return:
        """
        index_str = str(index)
        for preproc_node, demand in zip(["A", "B", "C", "D", "E"],
                                          [27.6, 26.7, 9.6, 3.6, 15.3]):
            req.add_node(preproc_node + index_str, demand=demand, ntype=self.universal_node_type)
        for sensor_actuator_node, demand in zip(["S", "T"], [2.0, 0.25]):
            req.add_node(sensor_actuator_node + index_str, demand=demand, ntype=self.universal_node_type, allowed_nodes=[substrate_node])
        # TODO: add edges

        return "E" + index_str, "D" + index_str, "T" + index_str


    def generate_request(self, name, raw_parameters, substrate):
        """
        Realizes the generator function to fit to the framework.

        :param name:
        :param raw_parameters:
        :param substrate:
        :return:
        """
        self._read_raw_parameters(raw_parameters)
        req = datamodel.Request("ABB_fog_app")
        nodes_for_actuators_sensors = sg.random.sample(substrate.nodes(), self.sensor_actuator_loop_count)
        for index in xrange(1, self.sensor_actuator_loop_count + 1):
            nodeE, nodeD, nodeT = self._add_single_preprocessing_block(req, index, nodes_for_actuators_sensors[index - 1])
        # TODO: connect edges

        return req


class ABBUseCaseFogNetworkGenerator(sg.ScenariogenerationTask):
    """
    Generates a industrial fog network described in
    Suter, Eidenbenz, Pignolet, Singla -- Fog Application Allocation for Automation Systems
    """

    def __init__(self, logger=None):
        super(ABBUseCaseFogNetworkGenerator).__init__(logger=logger)

    def apply(self, scenario_parameters, scenario):
        class_raw_parameters_dict = scenario_parameters[sg.SUBSTRATE_GENERATION_TASK].values()[0]
        class_name = self.__class__.__name__
        if class_name not in class_raw_parameters_dict:
            raise sg.ScenarioGeneratorError("No class name found in config file.")
        raw_parameters = class_raw_parameters_dict[class_name]
        substrate = datamodel.Substrate("ABB_fog_net")
        scenario.substrate = substrate