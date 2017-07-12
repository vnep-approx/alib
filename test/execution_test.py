from alib import run_experiment


class TestExecutionParameter:
    def setup(self):
        self.imaginary_execution_pspace = [
            {
                "ALGORITHM": {
                    "ID": "Alg1",
                    "ALGORITHM_PARAMETERS": {  # one algorithm with alg. & gurobi parameters
                        "alpha": [0.1, 0.2, 0.3],
                        "beta": [1, 2, 4]
                    },
                    "GUROBI_PARAMETERS": {
                        'timelimit': [1200, 1800],
                        'threads': [8]
                    }
                }
            },
            {
                "ALGORITHM": {
                    "ID": "Alg2",
                    "ALGORITHM_PARAMETERS": {  # one algorithm with alg. parameters only
                        "gamma": [0.1, 0.2, 0.3],
                    }
                }
            },
            {
                "ALGORITHM": {
                    "ID": "Alg3",
                    "GUROBI_PARAMETERS": {  # one algorithm with gurobi parameters only
                        'threads': [8]
                    }
                }
            },
            {
                "ALGORITHM": {
                    "ID": "Alg4"  # Duplicate name, no parameter values
                }
            }
        ]
        self.p_container = run_experiment.ExecutionParameters(self.imaginary_execution_pspace)

    def test_parameter_expansion_generates_correct_number(self):
        assert len(self.p_container.algorithm_parameter_list) == 0  # expecting 0
        self.p_container.generate_parameter_combinations()
        assert len(self.p_container.algorithm_parameter_list) == 23  # expecting 9 * 2 + 3 + 1 + 1 = 23

    def test_reverse_lookup(self):
        assert self.p_container.reverse_lookup == {}
        self.p_container.generate_parameter_combinations()
        assert len(self.p_container.reverse_lookup["Alg1"]["ALGORITHM_PARAMETERS"]["beta"][1]) == 6
        assert len(self.p_container.reverse_lookup["Alg1"]["GUROBI_PARAMETERS"]["timelimit"][1200]) == 9
        assert len(self.p_container.reverse_lookup["Alg2"]["ALGORITHM_PARAMETERS"]["gamma"][0.1]) == 1

    def test_parameter_expansion_contains_correct_parameters(self):
        p_container = run_experiment.ExecutionParameters(self.imaginary_execution_pspace)
        p_container.generate_parameter_combinations()
        for p in p_container.algorithm_parameter_list:
            assert "ALG_ID" in p
            assert "ALGORITHM_PARAMETERS" in p

    def test_reverse_look_up(self):
        p_container = run_experiment.ExecutionParameters(self.imaginary_execution_pspace)
        p_container.generate_parameter_combinations()
        set1 = p_container.get_execution_ids(ALG_ID="Alg1")
        assert set1 == set(range(0, 18))

        set2 = p_container.get_execution_ids(GUROBI_PARAMETERS={"timelimit": 1200})
        assert len(set2) == 9

        set2 = p_container.get_execution_ids(ALG_ID="Alg2")
        assert set2 == set(range(18, 21))

        set3 = p_container.get_execution_ids(ALG_ID="Alg3")
        assert set3 == set(range(21, 22))

        set4 = p_container.get_execution_ids(ALG_ID="Alg2", ALGORITHM_PARAMETERS={"gamma": 0.1})
        assert set4 == set([18])

        set4 = p_container.get_execution_ids(ALG_ID="Alg4")
        # self.assertEqual(set4, set(range(22, 23)))
