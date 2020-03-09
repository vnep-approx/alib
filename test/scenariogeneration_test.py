import cPickle as pickle
import copy
import os
import glob
import random
import time
import yaml

import numpy

from alib import scenariogeneration, datamodel, test_utils, util
import pytest

TEST_BASE_DIR = os.path.abspath(os.path.dirname(__file__))

_log_directory = None
@pytest.fixture(scope="session", autouse=True)
def check_and_create_log_diretory(request):
    print("\n\nChecking whether directory {} exists...".format(util.ExperimentPathHandler.LOG_DIR))
    if not os.path.exists(util.ExperimentPathHandler.LOG_DIR):
        print("\tdid not exist, will create...".format(util.ExperimentPathHandler.LOG_DIR))
        os.mkdir(util.ExperimentPathHandler.LOG_DIR)
        print("\tcreated.".format(util.ExperimentPathHandler.LOG_DIR))
        _log_directory = util.ExperimentPathHandler.LOG_DIR
        #only if it was created, we remove it...

        def remove_log_directory():
            if _log_directory is not None:
                import shutil
                print("\n\nGoing to remove directory {}..".format(_log_directory))
                for logfile in glob.glob(_log_directory + "/*.log"):
                    print("\tremoving file {}..".format(logfile))
                    os.remove(logfile)
                print("\tremoving directoy.")
                os.rmdir(_log_directory)
                print("\tOK.")

        request.addfinalizer(remove_log_directory)
    else:
        print("\tdirectory exists; will be reused!")



class TestChainRequestGenerator:
    def setup(self):
        self.base_parameters = {
            "topology": "Aarnet",
            "node_types": ("nt_1", "nt_2", "nt_3", "nt_4", "nt_5", "nt_6"),
            "node_cost_factor": 1.0,
            "node_capacity": 100.0,
            "edge_cost": 1.0,
            "edge_capacity": 100.0,
            "node_type_distribution": 1.0,
            "number_of_requests": 1,
            "min_number_of_nodes": 2,
            "max_number_of_nodes": 4,
            "probability": 0.3,
            "node_resource_factor": 0.5,
            "edge_resource_factor": 2.0
        }

        self.chain_gen = scenariogeneration.ServiceChainGenerator()
        self.substrate = scenariogeneration.TopologyZooReader().read_substrate(self.base_parameters)
        self.test_parameters_no_random_edges = dict(self.base_parameters)

        self.test_parameters_no_random_edges["max_number_of_nodes"] = 5
        self.test_parameters_no_random_edges["min_number_of_nodes"] = 5
        self.test_parameters_no_random_edges["probability"] = 0.0

        self.test_parameters_long_chain_no_random_edges = dict(self.base_parameters)
        self.test_parameters_long_chain_no_random_edges["max_number_of_nodes"] = 5
        self.test_parameters_long_chain_no_random_edges["min_number_of_nodes"] = 5

    def test_generates_linearrequest_when_probability_is_zero(self):
        req = self.chain_gen.generate_request("test_request",
                                              self.test_parameters_no_random_edges,
                                              self.substrate)
        assert isinstance(req, datamodel.LinearRequest), "Service Chain with probability 0.0 was not LinearRequest type!"

    def test_correct_number_of_edges_in_request(self):
        req = self.chain_gen.generate_request("test_request",
                                              self.test_parameters_no_random_edges,
                                              self.substrate)
        expected = self.test_parameters_no_random_edges["min_number_of_nodes"] + 1
        assert len(req.edges) == expected, "Wrong number of edges in request!"

    def test_chain_should_connect_source_with_target(self):
        req = self.chain_gen.generate_request("test_request",
                                              self.test_parameters_no_random_edges,
                                              self.substrate)

        # walk along the chain by following the out neighbors...
        current_node = scenariogeneration.ServiceChainGenerator.SOURCE_NODE
        for i in xrange(1 + self.test_parameters_no_random_edges["min_number_of_nodes"]):  # Should take |E| steps to get to the target node, and |E| = |V| + 1
            current_node = req.out_neighbors[current_node][0]  # no random edges => only 1 out neighbor

        assert current_node == scenariogeneration.ServiceChainGenerator.TARGET_NODE, "Did not reach target Node! \n{}".format(req.edges)

    def test_start_and_target_mapping_fixed_and_allowed_nodes_of_correct_type(self):
        sp = self.base_parameters
        sp["min_number_of_nodes"] = 10
        sp["max_number_of_nodes"] = 10
        sp["node_type_distribution"] = 0.5
        req = self.chain_gen.generate_request("test_request",
                                              sp,
                                              self.substrate)
        number_of_substrate_nodes = len(self.substrate.nodes)
        for req_node in req.get_nodes():
            allowed_nodes = req.node[req_node]["allowed_nodes"]

            # check that placement restrictions exist:
            assert allowed_nodes is not None, "node placement restriction are None!"
            number_of_allowed_nodes = len(allowed_nodes)
            assert number_of_allowed_nodes != 0, "No substrate nodes allowed for {}".format(req_node)

            # check the number of allowed nodes:
            if (req_node == scenariogeneration.ServiceChainGenerator.SOURCE_NODE
                    or req_node == scenariogeneration.ServiceChainGenerator.TARGET_NODE):
                assert number_of_allowed_nodes == 1, "Source and target should be fixed to a specific node"

            # check that each allowed node really allows the node type:
            req_node_type = req.node[req_node]["type"]
            supporting_substrate_nodes = self.substrate.get_nodes_by_type(req_node_type)
            for snode in allowed_nodes:
                supported_types = self.substrate.node[snode]["supported_types"]
                msg = "Request node {} of type {} was mapped to substrate node {}, which supports {}".format(req_node, req_node_type, snode, supported_types)
                assert snode in supporting_substrate_nodes, msg


class TestExponentialRequestGenerator:
    def setup(self):
        self.exp_gen = scenariogeneration.ExponentialRequestGenerator()
        self.base_parameters = {
            "topology": "Aarnet",
            "node_types": ("t1", "t2", "t3", "t4"),
            "node_cost_factor": 1.0,
            "node_capacity": 100.0,
            "edge_cost": 1.0,
            "edge_capacity": 100.0,
            "node_type_distribution": 1.0,
            "number_of_requests": 10,
            "min_number_of_nodes": 2,
            "max_number_of_nodes": 5,
            "probability": 0.5,
            "node_resource_factor": 0.5,
            "edge_resource_factor": 2.0,
            "potential_nodes_factor": 0.3
        }
        self.substrate = scenariogeneration.TopologyZooReader().read_substrate(self.base_parameters)

        self.base_parameters_large_request = dict(self.base_parameters)
        self.base_parameters["min_number_of_nodes"] = 30
        self.base_parameters["max_number_of_nodes"] = 30

        self.parameters_list_varying_probability = [dict(self.base_parameters), dict(self.base_parameters), dict(self.base_parameters)]
        self.parameters_list_varying_probability[0]["probability"] = 0.1
        self.parameters_list_varying_probability[1]["probability"] = 0.5
        self.parameters_list_varying_probability[2]["probability"] = 1.0

    def test_generate_request_returns_request_object(self):
        req = self.exp_gen.generate_request("test", self.base_parameters, self.substrate)
        assert not isinstance(req, datamodel.LinearRequest), "Exp. request generator should not produce a LinearRequest!"
        assert isinstance(req, datamodel.Request), "Exp. request generator should produce a Request!"

    def test_edge_count_scales_with_probability_within_ten_percent(self):
        n = self.parameters_list_varying_probability[0]["min_number_of_nodes"]
        possible_edges = n * (n - 1)
        req_list = [(sp["probability"], self.exp_gen.generate_request("test", sp, self.substrate))
                    for sp in self.parameters_list_varying_probability]
        for p, req in req_list:
            assert p * possible_edges == pytest.approx(req.get_number_of_edges(), abs=0.1 * possible_edges)


class TestUniformRequestGenerator:
    def setup(self):
        self.uni_gen = scenariogeneration.UniformRequestGenerator()
        self.test_substrate = test_utils.get_test_substrate(10)
        self.base_parameters = {
            "topology": "Aarnet",
            "node_types": ("t1", "t2", "t3", "t4"),
            "node_cost_factor": 1.0,
            "node_capacity": 100.0,
            "edge_cost": 1.0,
            "edge_capacity": 100.0,
            "node_type_distribution": 1.0,
            "number_of_requests": 20,
            "min_number_of_nodes": 2,
            "max_number_of_nodes": 5,
            "probability": 0.5,
            "variability": 0.2,
            "node_resource_factor": 0.5,
            "edge_resource_factor": 2.0,
            "potential_nodes_factor": 0.3
        }
        self.substrate = scenariogeneration.TopologyZooReader().read_substrate(self.base_parameters)

        req = self.uni_gen.generate_request("test", self.base_parameters, self.test_substrate)
        assert not isinstance(req, datamodel.LinearRequest), "Exp. request generator should not produce a LinearRequest!"
        assert isinstance(req, datamodel.Request), "Exp. request generator should produce a Request!"

    def test_edge_count_scales_with_probability_within_ten_percent(self):
        sp = self.base_parameters.copy()
        sp["number_of_requests"] = 50
        for n in range(15, 30, 5):
            possible_edges = n * (n - 1)
            for p in [0.3, 0.5, 0.8]:
                sp["min_number_of_nodes"] = n
                sp["max_number_of_nodes"] = n
                sp["probability"] = p

                avg_number_of_edges = sum(len(req.edges) for req in self.uni_gen.generate_request_list(sp, self.substrate)) / float(sp["number_of_requests"])
                expected = p * possible_edges
                assert expected == pytest.approx(avg_number_of_edges, abs=0.1 * expected)

    def test_fails_when_parameters_are_impossible(self):
        impossible_sp_node = self.base_parameters.copy()
        impossible_sp_node["node_resource_factor"] = 1000.0
        impossible_sp_node["edge_resource_factor"] = 1000.0  # remove edge resource bottleneck for this part of the test
        impossible_sp_node["number_of_requests"] = 1
        impossible_sp_node["max_number_of_nodes"] = 3
        impossible_sp_node["min_number_of_nodes"] = 3

        with pytest.raises(scenariogeneration.ScenarioParameterError) as e:
            self.uni_gen.generate_request("impossible", impossible_sp_node, self.substrate)
        assert "Parameters will always result in infeasible request due to demand for node type" in str(e.value)

        impossible_sp_edge = self.base_parameters.copy()
        impossible_sp_edge["edge_resource_factor"] = 0.001  # aim to use 1000 * substrate capacity
        impossible_sp_edge["node_resource_factor"] = 0.001  # remove node resource bottleneck for this part of the test
        impossible_sp_edge["number_of_requests"] = 1
        impossible_sp_edge["max_number_of_nodes"] = 3
        impossible_sp_edge["min_number_of_nodes"] = 3
        impossible_sp_edge["probability"] = 1.0
        with pytest.raises(scenariogeneration.ScenarioParameterError) as e:
            self.uni_gen.generate_request("impossible", impossible_sp_edge, self.substrate)
        assert "Parameters will always result in infeasible request due to edge demand" in str(e.value)


class TestCactusGenerator:
    def setup(self):
        self.req_gen = scenariogeneration.CactusRequestGenerator()
        self.base_parameters = {
            "topology": "Geant2012",
            "node_types": ("t1",),
            "node_cost_factor": 1.0,
            "node_capacity": 100.0,
            "edge_cost": 1.0,
            "edge_capacity": 100.0,
            "node_type_distribution": 0.5,
            "min_number_of_nodes": 3,
            "max_number_of_nodes": 3,
            "number_of_requests": 1,
            "iterations": 3,
            "max_cycles": 999,
            "layers": 4,
            "fix_root_mapping": True,
            "fix_leaf_mapping": True,
            "branching_distribution": (0.2, 0.4, 0.0, 0.0, 0.4),
            "probability": 1.0,
            "arbitrary_edge_orientations": False,
            "node_resource_factor": 0.05,
            "edge_resource_factor": 20.0,
            "potential_nodes_factor": 1.0
        }
        self.substrate = scenariogeneration.TopologyZooReader().read_substrate(self.base_parameters)

    def test_empirical_average_number_nodes_considers_min_max_number_nodes_limits(self):
        self.req_gen.generate_request_list(self.base_parameters, self.substrate)
        # Assert that the expected number of request nodes is equal to the fixed value
        # imposed by our choice of min_/max_number_of_nodes
        # print r_list
        assert self.req_gen._expected_number_of_request_nodes_per_type == pytest.approx(3)


class TestScenarioGenerator:
    def setup(self):
        # reapply the random seed, because these values should result in example scenarios that are computed relatively quickly
        numpy.random.seed(1234)
        scenariogeneration.random.seed("scenariogeneration")
        self.sg = scenariogeneration.ScenarioGenerator()
        self.sg2 = scenariogeneration.ScenarioGenerator()

    def test_reading_from_yaml_file_produces_correct_scenario_parameters(self):
        with open(os.path.join(TEST_BASE_DIR, "yaml/one_of_each.yml"), "r") as f:
            param_space = yaml.load(f)
        self.sg.generate_scenarios(param_space)
        assert len(self.sg.scenario_parameter_container.scenario_parameter_combination_list) == 16
        for scenario in self.sg.scenario_parameter_container.scenario_list:
            assert len(scenario.requests) == 5
            total_profit = sum(req.profit for req in scenario.requests)

    def test_scenario_repetition(self):
        with open(os.path.join(TEST_BASE_DIR, "yaml/one_of_each.yml"), "r") as f:
            param_space = yaml.load(f)
        x = self.sg.generate_scenarios(param_space, repetition=3)
        assert len(self.sg.scenario_parameter_container.scenario_parameter_combination_list) == 48
        assert len(self.sg.scenario_parameter_container.scenario_triple) == 48

    def test_scenario_id_offset(self):
        with open(os.path.join(TEST_BASE_DIR, "yaml/tinytiny.yml"), "r") as f:
            param_space = yaml.load(f)
        scenario_index_offset = 1337
        x = self.sg.generate_scenarios(param_space, repetition=3, scenario_index_offset=scenario_index_offset)
        num_scenarios = len(self.sg.scenario_parameter_container.scenario_parameter_combination_list)
        expected = set(range(scenario_index_offset, num_scenarios + scenario_index_offset))
        obtained = set(self.sg.scenario_parameter_container.scenario_triple.keys())
        assert obtained == expected

        scenario_ids_in_reverse_lookup = set()
        spd = self.sg.scenario_parameter_container.scenario_parameter_dict
        for task, strat_class_key_val_dict in spd.items():
            for strat, class_key_val_dict in strat_class_key_val_dict.items():
                all_set = set()
                for class_name, key_val_dict in class_key_val_dict.items():
                    if class_name == "all":
                        continue
                    for key, val_dict in key_val_dict.items():
                        for val, id_set in val_dict.items():
                            all_set |= id_set
                            scenario_ids_in_reverse_lookup |= id_set
                assert all_set == spd[task][strat]["all"]
                assert len(all_set) != 0
        assert scenario_ids_in_reverse_lookup == expected

    def test_merge_scenario_parameter_container(self):
        with open(os.path.join(TEST_BASE_DIR, "yaml/tinytiny.yml"), "r") as f:
            param_space = yaml.load(f)
        with open(os.path.join(TEST_BASE_DIR, "yaml/one_of_each.yml"), "r") as f:
            param_space_2 = yaml.load(f)
        self.sg.generate_scenarios(param_space, repetition=3, scenario_index_offset=0)
        spc_base = self.sg.scenario_parameter_container
        offset = len(spc_base.scenario_list)
        self.sg2.generate_scenarios(param_space_2, repetition=2, scenario_index_offset=offset)
        spc_other = self.sg2.scenario_parameter_container

        spc_base_pre_merge = copy.deepcopy(spc_base)
        spc_other_pre_merge = copy.deepcopy(spc_other)

        def comparable_scenarios(scenario_list):
            result = []
            for s in scenario_list:
                reqs = [(r.name, r.nodes, r.edges) for r in s.requests]
                result.append((s.name, s.substrate.name, reqs, s.objective))
            return result

        spc_base.merge_with_other_scenario_parameter_container(spc_other)
        obtained = comparable_scenarios(spc_base.scenario_list)
        expected = comparable_scenarios(spc_base_pre_merge.scenario_list + spc_other_pre_merge.scenario_list)
        assert len(expected) == len(obtained)
        assert expected == obtained

        expected_ids = set(range(0, len(spc_base_pre_merge.scenario_list) + len(spc_other_pre_merge.scenario_list)))
        obtained_ids = set(spc_base.scenario_triple.keys())
        assert expected_ids == obtained_ids

        spd_base = copy.deepcopy(spc_base_pre_merge.scenario_parameter_dict)
        spd_other = copy.deepcopy(spc_other_pre_merge.scenario_parameter_dict)
        spd_merged = spc_base.scenario_parameter_dict
        for task, strat_class_key_val_dict in spd_merged.items():
            assert (task in spd_base) or (task in spd_other)
            base_strat_class_key_val_dict = spd_base.get(task, {})
            other_strat_class_key_val_dict = spd_other.get(task, {})
            for strat, class_key_val_dict in strat_class_key_val_dict.items():
                assert (strat in base_strat_class_key_val_dict) or (strat in other_strat_class_key_val_dict)
                base_class_key_val_dict = base_strat_class_key_val_dict.get(strat, {})
                other_class_key_val_dict = other_strat_class_key_val_dict.get(strat, {})
                assert spd_merged[task][strat]["all"] == base_class_key_val_dict.get("all", set()) | other_class_key_val_dict.get("all", set())
                for class_name, key_val_dict in class_key_val_dict.items():
                    if class_name == "all":
                        continue
                    assert (class_name in base_class_key_val_dict) or (class_name in other_class_key_val_dict)
                    base_key_val_dict = base_strat_class_key_val_dict.get(class_name, {})
                    other_key_val_dict = other_strat_class_key_val_dict.get(class_name, {})
                    for key, val_dict in key_val_dict.items():
                        base_val_dict = base_class_key_val_dict.get(class_name, {}).get(key, {})
                        other_val_dict = other_class_key_val_dict.get(class_name, {}).get(key, {})
                        for val, id_set in val_dict.items():
                            assert id_set == base_val_dict.get(val, set()) | other_val_dict.get(val, set())
                            base_val_dict.pop(val, None)
                            other_val_dict.pop(val, None)
                        assert not base_val_dict
                        assert not other_val_dict
                        base_key_val_dict.pop(key, None)
                        other_key_val_dict.pop(key, None)
                    assert not base_key_val_dict
                    assert not other_key_val_dict
                    base_class_key_val_dict.pop(class_name, None)
                    other_class_key_val_dict.pop(class_name, None)
                    base_class_key_val_dict.pop("all", None)
                    other_class_key_val_dict.pop("all", None)
                base_strat_class_key_val_dict.pop(strat, None)
                other_strat_class_key_val_dict.pop(strat, None)
            assert not base_strat_class_key_val_dict
            assert not other_strat_class_key_val_dict

        # Check that "other" is not modified
        assert comparable_scenarios(spc_other_pre_merge.scenario_list) == comparable_scenarios(spc_other.scenario_list)
        assert spc_other_pre_merge.scenario_parameter_combination_list == spc_other.scenario_parameter_combination_list
        assert spc_other_pre_merge.scenario_parameter_dict == spc_other.scenario_parameter_dict
        assert spc_other_pre_merge.scenario_triple.keys() == spc_other.scenario_triple.keys()

    def test_tiny_yaml_file_produces_correct_scenario_para_container_dict(self):
        with open(os.path.join(TEST_BASE_DIR, "yaml/tinytiny.yml"), "r") as f:
            param_space = yaml.load(f)
        self.sg.generate_scenarios(param_space)
        scp = self.sg.scenario_parameter_container.scenario_parameter_dict
        assert {0} == scp['node_placement_restriction_mapping']['neighbors']['NeighborhoodSearchRestrictionGenerator']['potential_nodes_factor'][0.3]
        assert {0} == scp['substrate_generation']['substrates']['TopologyZooReader']['node_type_distribution'][0.3]
        assert {0} == scp['node_placement_restriction_mapping']['neighbors']['all']

    def test_missing_required_parameters_or_generation_tasks_cause_appropriate_exceptions_and_warnings(self):
        with open(os.path.join(TEST_BASE_DIR, "yaml/tinytiny.yml"), "r") as f:
            param_space = yaml.load(f)

        # Remove a required task:
        del param_space[scenariogeneration.REQUEST_GENERATION_TASK]
        # Remove an optional task:
        del param_space[scenariogeneration.PROFIT_CALCULATION_TASK]
        # Remove some required parameters:
        del param_space[scenariogeneration.SUBSTRATE_GENERATION_TASK][0]["substrates"]["TopologyZooReader"]["topology"]
        del param_space[scenariogeneration.SUBSTRATE_GENERATION_TASK][0]["substrates"]["TopologyZooReader"]["node_types"]
        # Add some unnecessary parameters:
        param_space[scenariogeneration.SUBSTRATE_GENERATION_TASK][0]["substrates"]["TopologyZooReader"]["additional_1"] = [5]
        param_space[scenariogeneration.SUBSTRATE_GENERATION_TASK][0]["substrates"]["TopologyZooReader"]["additional_2"] = [5]
        # Add a strategy with an arbitrary class to check what happens when a class does not define expected params:
        param_space[scenariogeneration.NODE_PLACEMENT_TASK].append({"fake_strategy": {"RequestGenerationError": {}}})

        with pytest.raises(scenariogeneration.ExperimentSpecificationError):
            scenariogeneration.verify_completeness_of_scenario_parameters(param_space)

    def test_tiny_yaml_file_produces_pickle(self):
        currDir = os.path.dirname(os.path.realpath(__file__))
        util.ExperimentPathHandler.OUTPUT_DIR = currDir
        with open(os.path.join(TEST_BASE_DIR, "yaml/tinytiny.yml"), "r") as f:
            scenariogeneration.generate_pickle_from_yml(f, "test.pickle")
        out = os.path.abspath(os.path.join(util.ExperimentPathHandler.OUTPUT_DIR, "test.pickle"))
        assert os.path.exists(out)
        param_dict = pickle.load(open(out, "rb"))
        with open(os.path.join(TEST_BASE_DIR, "yaml/tinytiny.yml"), "r") as f:
            param_space = yaml.load(f)
        self.sg.generate_scenarios(param_space)
        container = self.sg.scenario_parameter_container
        assert len(param_dict.scenario_list) == len(container.scenario_list)

    def test_mp_scenario_generation(self):
        with open(os.path.join(TEST_BASE_DIR, "yaml/tinytiny.yml"), "r") as f:
            param_space = yaml.load(f)
        mp_start = time.time()
        self.sg.threads = 2
        triple_mp = self.sg.generate_scenarios(param_space)
        mp_end = time.time()
        mp_diff = mp_end - mp_start
        sp_start = time.time()
        triple_sp = self.sg2.generate_scenarios(param_space)
        sp_end = time.time()
        sp_diff = sp_end - sp_start

        assert len(triple_sp) == len(triple_mp)

        for triple in triple_sp.items():
            (index, (sp, scenario)) = triple
            (sp_sp, scenario_sp) = triple_sp[index]
            assert sp_sp == sp
            assert scenario.name == scenario_sp.name
            assert scenario.substrate == scenario_sp.substrate
            for req_i, req in enumerate(scenario.requests):
                assert req == scenario_sp.requests[req_i]
        assert self.sg2.scenario_parameter_container.scenario_parameter_dict == self.sg.scenario_parameter_container.scenario_parameter_dict


class TestRequestGeneration:
    def setup(self):
        self.cactus = scenariogeneration.CactusRequestGenerator()
        self.exp = scenariogeneration.ExponentialRequestGenerator()
        self.chains = scenariogeneration.ServiceChainGenerator()
        self.uniform = scenariogeneration.UniformRequestGenerator()
        self.req_gens = [
            self.cactus,
            self.exp,
            self.chains,
            self.uniform
        ]
        base = {
            "number_of_requests": 20,
            "min_number_of_nodes": 4,
            "max_number_of_nodes": 8,
            "node_resource_factor": 0.001,
            "edge_resource_factor": 1000.0
        }
        self.params = {}
        self.params[self.cactus] = dict(base.items() + {
            "iterations": 300,  # accurate number of edges estimate is important for this test!
            "max_cycles": 20,
            "layers": 3,
            "fix_root_mapping": True,
            "fix_leaf_mapping": True,
            "branching_distribution": (0.0, 0.8, 0.1, 0.1),
            "probability": 1.0,
            "arbitrary_edge_orientations": False,
        }.items())
        self.params[self.exp] = dict(base.items() + {
            "probability": 1.0
        }.items())
        self.params[self.uniform] = dict(base.items() + {
            "probability": 1.0,
            "variability": 0.8
        }.items())
        self.params[self.chains] = dict(base.items() + {
            "probability": 1.0
        }.items())

        self.substrate = scenariogeneration.TopologyZooReader().read_substrate({
            "topology": "Geant2012",
            "node_types": ("t1", "t2", "t3"),
            "node_cost_factor": 1.0,
            "node_capacity": 100.0,
            "edge_cost": 1.0,
            "edge_capacity": 100.0,
            "node_type_distribution": 0.5
        })

    def test_used_resources_match_edge_and_node_resource_factor(self):
        number_of_scenarios = 1
        number_of_requests = 10
        tolerance = 0.0001
        prob = 0.5
        node_res = random.uniform(0.1, 0.8)
        edge_res = random.uniform(2.0, 20.0)

        for req_gen in self.req_gens:
            scenariogeneration.random.seed(random.random())
            sp = self.params[req_gen]
            sp["number_of_requests"] = number_of_requests
            sp["node_resource_factor"] = node_res
            sp["edge_resource_factor"] = edge_res
            sp["probability"] = prob
            requests = []
            for _ in xrange(number_of_scenarios):
                # TODO: why are the edge resources in the unnormalized resource allocation 1-2% greater than expected??
                requests += req_gen.generate_request_list(sp, self.substrate, normalize=True)
            used_node_resources = {nt: 0.0 for nt in self.substrate.types}
            available_node_resources = {nt: self.substrate.get_total_node_resources(nt)
                                        for nt in self.substrate.types}
            available_edge_res = self.substrate.get_total_edge_resources()
            used_edge_resources = sum(req.edge[e]["demand"] for req in requests for e in req.edges) / float(number_of_scenarios)
            for req in requests:
                for node in req.nodes:
                    nt = req.node[node]["type"]
                    demand = req.node[node]["demand"]
                    used_node_resources[nt] += demand / float(number_of_scenarios)
            for nt in sorted(self.substrate.types):
                used_ratio = used_node_resources[nt] / available_node_resources[nt]
                msg = "Expected node resource use around {}, but found {}".format(node_res, used_ratio)
                assert node_res == pytest.approx(used_ratio, abs=tolerance * node_res), msg
            edge_usage = used_edge_resources / available_edge_res
            assert 1.0 / edge_res == pytest.approx(edge_usage, abs=tolerance / edge_res)  # msg="Expected edge resource use around {}, but found {}".format(1.0 / edge_res, edge_usage))

    def test_generate_request_list_generates_correct_number_of_requests(self):
        for rg in self.req_gens:
            parameters = self.params[rg]
            parameters["node_resource_factor"] = 0.001
            parameters["edge_resource_factor"] = 1000
            requests = rg.generate_request_list(parameters, self.substrate, base_name="test_{id}")
            assert len(requests) == parameters["number_of_requests"], "Request dictionary had wrong number of vnets!"

    def test_node_types_are_assigned(self):
        for req_gen in self.req_gens:
            parameters = self.params[req_gen]
            parameters["node_resource_factor"] = 0.001
            parameters["edge_resource_factor"] = 1000.0
            req_list = req_gen.generate_request_list(parameters, self.substrate)
            present_functions = set()
            types = self.substrate.types
            for req in req_list:
                for i in req.nodes:
                    ntype = req.node[i]["type"]
                    assert ntype in types
                    present_functions.add(ntype)
            for nt in types:
                assert nt in present_functions

    def test_generate_correct_number_of_nodes(self):
        for req_gen in self.req_gens:
            parameters = self.params[req_gen]
            parameters["node_resource_factor"] = 0.001
            parameters["edge_resource_factor"] = 1000.0
            req_list = req_gen.generate_request_list(parameters, self.substrate)
            min_expected = parameters["min_number_of_nodes"]
            max_expected = parameters["max_number_of_nodes"]
            if req_gen == self.chains:  # chain generator does not count source & target nodes
                min_expected += 2
                max_expected += 2
            for r in req_list:
                assert min_expected <= len(r.nodes)
                assert len(r.nodes) <= max_expected


class TestNodePlacementRestriction:
    def setup(self):
        self.chain_gen = scenariogeneration.ServiceChainGenerator()
        self.parameters = {
            "node_types": ("t1", "t2"),
            "topology": "Aarnet",
            "node_cost_factor": 1.0,
            "node_capacity": 100.0,
            "edge_cost": 1.0,
            "edge_capacity": 100.0,
            "number_of_requests": 10,
            "min_number_of_nodes": 3,
            "max_number_of_nodes": 7,
            "variability": 0.0,
            "node_type_distribution": 0.4,
            "potential_nodes_factor": 0.8,
            "probability": 0.5,
            "node_resource_factor": 0.5,
            "edge_resource_factor": 2.0,
            "profit_factor": 1.5,
            "iterations": 10,
            "max_cycles": 20,
            "layers": 4,
            "arbitrary_edge_orientations": False,
            "fix_root_mapping": True,
            "fix_leaf_mapping": True,
            "branching_distribution": (0.0, 0.5, 0.3, 0.15, 0.05)
        }
        self.substrate = scenariogeneration.TopologyZooReader().read_substrate(self.parameters)

    def test_placement_restrictions_work_with_all_combinations_of_request_and_restriction_generators(self):
        restriction_generator = scenariogeneration.NeighborhoodSearchRestrictionGenerator()
        self._verify_that_placement_restrictions_are_generated_by_restriction_generator(restriction_generator, True)
        restriction_generator = scenariogeneration.UniformEmbeddingRestrictionGenerator()
        self._verify_that_placement_restrictions_are_generated_by_restriction_generator(restriction_generator, False)

    def test_fix_root_leaf_mapping(self):
        cactus = scenariogeneration.CactusRequestGenerator()
        requests = cactus.generate_request_list(self.parameters, self.substrate)
        restriction_generators = [scenariogeneration.NeighborhoodSearchRestrictionGenerator(),
                                  scenariogeneration.UniformEmbeddingRestrictionGenerator()]
        for req in requests:
            for restriction_generator in restriction_generators:
                req_copy = copy.deepcopy(req)
                restriction_generator.generate_restrictions_single_request(
                    req_copy,
                    self.substrate,
                    self.parameters
                )
                fixed_nodes = [i for i in req_copy.nodes
                               if len(req_copy.get_allowed_nodes(i)) == 1]
                assert len(fixed_nodes) != 0
        self.parameters["fix_root_mapping"] = False
        self.parameters["fix_leaf_mapping"] = False
        requests = cactus.generate_request_list(self.parameters, self.substrate)
        for req in requests:
            for restriction_generator in restriction_generators:
                req_copy = copy.deepcopy(req)
                restriction_generator.generate_restrictions_single_request(
                    req_copy,
                    self.substrate,
                    self.parameters
                )
                fixed_nodes = [i for i in req_copy.nodes
                               if len(req_copy.get_allowed_nodes(i)) == 1]
                assert len(fixed_nodes) == 0

    def _verify_that_placement_restrictions_are_generated_by_restriction_generator(self,
                                                                                   restriction_generator,
                                                                                   number_of_allowed_nodes_not_constant):
        # Generate some scenarios using each type of request generator:
        req_generators = dict(
            cactus=scenariogeneration.CactusRequestGenerator(),
            uniform=scenariogeneration.UniformRequestGenerator(),
            chain=scenariogeneration.ServiceChainGenerator(),
            exponential=scenariogeneration.ExponentialRequestGenerator()
        )
        scenarios = {}
        for request_type, req_generator in req_generators.iteritems():
            requests = req_generator.generate_request_list(self.parameters, self.substrate)
            scenarios[request_type] = datamodel.Scenario("test_{}".format(request_type), self.substrate, requests)

        for request_type, scenario in scenarios.iteritems():
            # TODO: Update this when fixed nodes are optional for cactus graph:
            may_have_fixed_nodes = request_type in ["chain", "cactus"]  # chain type must have fixed start/end node to generate latency constraint...
            expected_allowed_node_count_before_restrictions = int(
                self.parameters["node_type_distribution"] * len(scenario.substrate.nodes)
            )
            # verify that the request generator does not impose node placement restrictions,
            # and allows each request node on all substrate nodes of same type
            self._verify_that_placement_restrictions_exist(scenario,
                                                           expected_allowed_node_count_before_restrictions,
                                                           allow_fixed_nodes=may_have_fixed_nodes)

            # Now, place restrictions on the node mapping
            restriction_generator.generate_and_apply_restrictions(scenario, self.parameters)
            # And verify that each node has the expected number of allowed nodes!
            if number_of_allowed_nodes_not_constant:
                expected_allowed_node_count_after_restrictions = None
            else:
                expected_allowed_node_count_after_restrictions = int(expected_allowed_node_count_before_restrictions * self.parameters["potential_nodes_factor"])
            self._verify_that_placement_restrictions_exist(scenario, expected_allowed_node_count_after_restrictions, allow_fixed_nodes=may_have_fixed_nodes)

    def _verify_that_placement_restrictions_exist(self, scenario, expected_number_of_allowed_nodes=None, allow_fixed_nodes=False):
        for req in scenario.requests:
            for node in req.nodes:
                assert "allowed_nodes" in req.node[node]
                allowed_nodes = req.node[node]["allowed_nodes"]
                assert allowed_nodes is not None
                ntype = req.node[node]["type"]
                substrate_nodes_with_correct_type = scenario.substrate.get_nodes_by_type(ntype)
                if expected_number_of_allowed_nodes is not None:
                    if allow_fixed_nodes and len(allowed_nodes) == 1:
                        continue
                    else:
                        assert len(allowed_nodes) == expected_number_of_allowed_nodes
                # verify that nodes have correct type!
                for u in allowed_nodes:
                    assert u in substrate_nodes_with_correct_type


class TestRandomEmbeddingProfitCalculator:
    def setup(self):
        self.profit_calculator = scenariogeneration.RandomEmbeddingProfitCalculator()

    def test_profit_scales_with_profit_factor(self):
        test_scenario = test_utils.get_test_scenario(number_of_requests=1, substrate_size=3)
        req = test_scenario.requests[0]
        sp1 = {"profit_factor": 1.0, "iterations": 1000}
        self.profit_calculator.generate_and_apply_profits(test_scenario, sp1)
        p1 = req.profit
        sp2 = {"profit_factor": 2.0, "iterations": 1000}

        # sp2 = test_utils.rst.get_test_scenario_parameters(profit_factor=2.0)
        self.profit_calculator.generate_and_apply_profits(test_scenario, sp2)
        p2 = req.profit
        sp3 = {"profit_factor": 3.0, "iterations": 1000}

        self.profit_calculator.generate_and_apply_profits(test_scenario, sp3)
        p3 = req.profit
        assert p2 == pytest.approx(2 * p1, rel=0.05), "Profit should double with doubled profit_factor"
        assert p3 == pytest.approx(3 * p1, rel=0.05), "Profit should triple with tripled profit_factor"

    def test_profit_scales_with_request_demand(self):
        scenario_1 = test_utils.get_test_scenario(request_demand=1)
        scenario_2 = test_utils.get_test_scenario(request_demand=2)
        scenario_3 = test_utils.get_test_scenario(request_demand=3)
        scenario_parameters = {"profit_factor": 1.0, "iterations": 1000}
        self.profit_calculator.generate_and_apply_profits(scenario_1, scenario_parameters)
        req = scenario_1.requests[0]
        p1 = req.profit
        self.profit_calculator.generate_and_apply_profits(scenario_2, scenario_parameters)
        req = scenario_2.requests[0]
        p2 = req.profit
        self.profit_calculator.generate_and_apply_profits(scenario_3, scenario_parameters)
        req = scenario_3.requests[0]
        p3 = req.profit
        assert p2 == pytest.approx(2 * p1, rel=0.05), "Profit should double with doubled request demand"
        assert p3 == pytest.approx(3 * p1, rel=0.05), "Profit should triple with tripled request demand"

    def test_can_calculate_profits_in_real_scenario(self):
        chain_gen = scenariogeneration.ServiceChainGenerator()
        raw_parameters = {
            "node_types": ("universal",),
            "topology": "Aarnet",
            "node_cost_factor": 1.0,
            "node_capacity": 100.0,
            "edge_cost": 1.0,
            "edge_capacity": 100.0,
            "number_of_requests": 20,
            "min_number_of_nodes": 3,
            "max_number_of_nodes": 7,
            "node_type_distribution": 0.3,
            "probability": 0.0,
            "node_resource_factor": 0.5,
            "edge_resource_factor": 2.0,
            "profit_factor": 1.5,
            "iterations": 5
        }
        substrate = scenariogeneration.TopologyZooReader().read_substrate(raw_parameters)
        requests = chain_gen.generate_request_list(raw_parameters, substrate, normalize=True)
        real_scenario = datamodel.Scenario("test", substrate, requests)
        self.profit_calculator.generate_and_apply_profits(real_scenario, raw_parameters)
        total_profits = sum(req.profit for req in real_scenario.requests)
        for req in requests:
            assert "profit_calculation_time" in req.graph
            assert req.graph["profit_calculation_time"] > 0
        assert total_profits > 0.1, "Profit should be non-zero"


class TestOptimalEmbeddingProfitCalculator:
    def setup(self):
        # scenariogeneration.random.seed(3)
        self.chain_gen = scenariogeneration.ServiceChainGenerator()
        self.parameters = {
            "number_of_requests": 3,
            "topology": "Aarnet",
            "node_types": ("t1", "t2"),
            "node_cost_factor": 1.0,
            "node_capacity": 100.0,
            "edge_cost": 1.0,
            "edge_capacity": 100.0,
            "min_number_of_nodes": 3,
            "max_number_of_nodes": 6,
            "node_type_distribution": 0.3,
            "probability": 0.0,
            "node_resource_factor": 0.02,
            "edge_resource_factor": 50.0,
            "profit_factor": 1,
            "timelimit": 20
        }
        self.substrate = scenariogeneration.TopologyZooReader().read_substrate(self.parameters)
        self.requests = self.chain_gen.generate_request_list(self.parameters, self.substrate, normalize=True)
        self.scenario = datamodel.Scenario("test_scen", self.substrate, self.requests)
        self.pc = scenariogeneration.OptimalEmbeddingProfitCalculator()

    def test_profit_for_real_scenario_should_be_nonzero(self):
        self.pc.generate_and_apply_profits(self.scenario, self.parameters)
        total_profit = sum(r.profit for r in self.scenario.requests)
        for req in self.scenario.requests:
            assert "profit_calculation_time" in req.graph
            assert req.graph["profit_calculation_time"] > 0
        assert total_profit > 0.0001, "Should generate non-zero profit"  # this should work once the min-cost objective is implemented in the ModelCreator

    def test_profit_for_artificial_scenario_should_be_correct(self):
        test_sub = datamodel.Substrate("test_sub")
        test_sub.add_node("s1", ["t1"], capacity={"t1": 2.001}, cost={"t1": 1.0})
        test_sub.add_node("s2", ["t2"], {"t2": 1.001}, {"t2": 1.0})
        test_sub.add_edge("s1", "s2", capacity=1.001, bidirected=True)
        test_req = datamodel.Request("test_req")
        # Request with 3 nodes, demand 1, 0.5, 0.5
        test_req.add_node("r1", 1.0, "t1", allowed_nodes=["s1"])
        test_req.add_node("r2", 0.3, "t2", allowed_nodes=["s2"])
        test_req.add_node("r3", 0.3, "t1", allowed_nodes=["s1"])
        test_req.add_edge("r1", "r2", 0.5)
        test_req.add_edge("r2", "r3", 0.5)
        scenario = datamodel.Scenario("foo", test_sub, [test_req], objective=datamodel.Objective.MAX_PROFIT)
        self.pc.generate_and_apply_profits(scenario, self.parameters)
        profit = test_req.profit
        assert profit == pytest.approx(2.6)  # this should work once the min-cost objective is implemented in the ModelCreator

    def test_infeasible_request_has_zero_profit(self):
        test_sub = datamodel.Substrate("test_sub")
        test_sub.add_node("s1", ["t1"], capacity={"t1": 1.0}, cost={"t1": 1.0})
        test_sub.add_node("s2", ["t2"], {"t2": 1.00}, {"t2": 1.0})
        test_sub.add_edge("s1", "s2", capacity=1.00, bidirected=True)

        test_req = datamodel.Request("test_req")
        test_req.add_node("r1", 100000.0, "t1", allowed_nodes=["s1", "s2"])
        test_req.add_node("r2", 0.3, "t2", allowed_nodes=["s1", "s2"])
        test_req.add_node("r3", 0.3, "t1", allowed_nodes=["s1", "s2"])
        test_req.add_edge("r1", "r2", 0.5)
        test_req.add_edge("r2", "r3", 0.5)

        scenario = datamodel.Scenario("foo", test_sub, [test_req], objective=datamodel.Objective.MAX_PROFIT)
        self.pc.generate_and_apply_profits(scenario, self.parameters)
        profit = test_req.profit
        assert profit == pytest.approx(0.0)


class TestTopologyZooReader:
    def setup(self):
        geant_parameters = {"node_types": ("t1", "t2"),
                            "topology": "Geant2012",
                            "edge_capacity": 100.0,
                            "node_capacity": 100.0,
                            "node_cost_factor": 1.0,
                            "node_type_distribution": 1.0}
        bell_parameters = {"node_types": ("t1", "t2"),
                           "topology": "Bellcanada",
                           "edge_capacity": 100.0,
                           "node_capacity": 100.0,
                           "node_cost_factor": 1.0,
                           "node_type_distribution": 1.0}
        self.top_zoo_reader = scenariogeneration.TopologyZooReader()
        self.substrate_geant = self.top_zoo_reader.read_substrate(geant_parameters)
        self.substrate_bell = self.top_zoo_reader.read_substrate(bell_parameters)

    def test_reader_produces_substrate_object(self):
        assert isinstance(self.substrate_geant, datamodel.Substrate), "Reader did not create an instance of Substrate!"
        assert self.substrate_geant.get_name() == "Geant2012"
        assert self.substrate_bell.get_name() == "Bellcanada"

    def test_substrate_has_correct_number_of_nodes_and_edges(self):
        # determined correct reference node & edge count manually by searching the Geant2012.gml file for "node [" and "edge ["
        node_count = len(self.substrate_geant.nodes)
        assert node_count == 40, "Geant2012 should have 40 nodes, but has {}".format(node_count)
        edge_count = len(self.substrate_geant.edges)
        assert edge_count == 122, "Geant2012 should have 122 (directed) edges, but has {}".format(edge_count)
        node_count = len(self.substrate_bell.nodes)
        assert node_count == 48, "Bellcanada should have 48 nodes, but has {}".format(node_count)
        edge_count = len(self.substrate_bell.edges)
        assert edge_count == 128, "Bellcanada should have 128 (directed) edges, but has {}".format(edge_count)

    def test_substrate_has_correct_costs_and_capacities(self):
        parameters = {
            "node_types": (scenariogeneration.UNIVERSAL_NODE_TYPE,),
            "topology": "Bellcanada",
            "node_capacity": 73.0,
            "node_cost_factor": 893.0,
            "edge_capacity": 1234.0,
            "node_type_distribution": 1.0
        }
        self.top_zoo_reader = scenariogeneration.TopologyZooReader()
        sub = self.top_zoo_reader.read_substrate(parameters)

        # check node attributes:
        assert sub.average_node_capacity(scenariogeneration.UNIVERSAL_NODE_TYPE) == pytest.approx(parameters["node_capacity"]), "Average node capacity mismatch"

        for node in sub.nodes:
            cap = sub.get_node_type_capacity(node, scenariogeneration.UNIVERSAL_NODE_TYPE)
            assert cap == parameters["node_capacity"], "Node capacity {} did not match expected {}!".format(cap, parameters["node_capacity"])
            ##THIS TEST DOESNT WORK AS THE MEANING OF NODE_COST has changed (now a factor)..

            # cost = sub.get_node_type_cost(node, scenariogeneration.UNIVERSAL_NODE_TYPE)
            # self.assertEqual(cost, parameters["node_cost_factor"], "Node cost {} did not match expected {}!".format(cost, parameters["node_cost_factor"]))

        # check edge attributes:
        assert sub.average_edge_capacity() == pytest.approx(parameters["edge_capacity"]), "Average edge capacity mismatch"
        for edge in sub.edges:
            cap = sub.get_edge_capacity(edge)
            assert cap == parameters["edge_capacity"], "Edge capacity {} did not match expected {}!".format(cap, parameters["edge_capacity"])


class TestSubstrateTransformation:
    def setup(self):
        self.parameters = {
            "node_types": ["t1", "t2"],
            "topology": "Aarnet",
            "node_cost_factor": 1.0,
            "node_capacity": 100.0,
            "edge_cost": 1.0,
            "edge_capacity": 100.0,
            "number_of_requests": 1,
            "min_number_of_nodes": 3,
            "max_number_of_nodes": 6,
            "node_type_distribution": 0.2,
            "probability": 0.0,
            "node_resource_factor": 0.02,
            "edge_resource_factor": 50.0,
            "profit_factor": 1,
        }
        self.sub_reader = scenariogeneration.TopologyZooReader()
        self.sub = self.sub_reader.read_substrate(self.parameters)

    def test_only_defined_nodes(self):
        ntypes = self.parameters["node_types"]
        for u in self.sub.nodes:
            for ntype in self.sub.node[u]["supported_types"]:
                msg = "Only the defined node types {} should exist, found {}".format(", ".join(ntypes), ntype)
                assert ntype in ntypes, msg

    def test_node_types_distributed_evenly(self):
        expected = int(self.parameters["node_type_distribution"] * len(self.sub.nodes))
        for nt in self.parameters["node_types"]:
            number_of_supporting_nodes = len(self.sub.get_nodes_by_type(nt))
            assert expected == number_of_supporting_nodes
