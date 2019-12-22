import timeit

from alib import util, modelcreator


class TestTemporalLog:
    def setup(self):
        self.tl_low_resolution = modelcreator.TemporalLog(log_interval_in_seconds=10)
        self.tl_low_resolution.set_global_start_time(timeit.default_timer())

        self.temp_log = modelcreator.TemporalLog(log_interval_in_seconds=0.00001)
        self.temp_log.set_global_start_time(timeit.default_timer())

        self.solution_status = modelcreator.GurobiStatus(status=modelcreator.GurobiStatus.OPTIMAL,
                                                         solCount=2,
                                                         objValue=2.0,
                                                         objBound=100.0,
                                                         objGap=util.get_obj_gap(2.0, 100.0),
                                                         integralSolution=True)

    def test_add_log_entry(self):
        data = modelcreator.MIPData(10, 1.0, 100.0, 1, -1)
        self.temp_log.add_log_data(data, 1.0)

        number_of_improved_entries = len(self.temp_log.improved_entries)
        number_of_entries = len(self.temp_log.log_entries)
        assert number_of_improved_entries == 1
        assert number_of_entries == 1

        last_entry = self.temp_log.log_entries[-1]
        assert last_entry.data == data

    def test_force_new(self):
        data_new = modelcreator.MIPData(10, 50.0, 50.0, 1, -1)
        self.tl_low_resolution.add_log_data(data_new, 1.1)

        data_new = modelcreator.MIPData(10, 50.0, 50.0, 1, -1)
        self.tl_low_resolution.add_log_data(data_new, 1.2, force_new_entry=True)

        assert len(self.tl_low_resolution.log_entries) == 2
        assert len(self.tl_low_resolution.improved_entries) == 1

    def test_replace_entry_when_a_better_one_is_added_quickly(self):
        data_old = modelcreator.MIPData(10, 1.0, 100.0, 1, -1)
        self.tl_low_resolution.add_log_data(data_old, 2.0)

        data_new = modelcreator.MIPData(20, 2.0, 50.0, 2, -1)  # better objective, better bound
        self.tl_low_resolution.add_log_data(data_new, 2.1)  # added 0.1 s after first

        number_of_entries = len(self.tl_low_resolution.log_entries)
        number_of_improved_entries = len(self.tl_low_resolution.improved_entries)
        assert number_of_entries == 1
        assert number_of_improved_entries == 1

        last_entry = self.tl_low_resolution.log_entries[-1]
        assert last_entry.data == data_new
        assert last_entry.data.objective_value == 2.0
        assert last_entry.data != data_old

        last_improved_entry = self.tl_low_resolution.improved_entries[-1]
        assert last_improved_entry.data == data_new
        assert last_improved_entry.data != data_old

    def test_replacing_entry_is_added_to_improved_entries_if_last_was_improved(self):
        data_old = modelcreator.MIPData(10, 2.0, 100.0, 1, -1)
        self.tl_low_resolution.add_log_data(data_old, 2.0)

        number_of_entries = len(self.tl_low_resolution.log_entries)
        number_of_improved_entries = len(self.tl_low_resolution.improved_entries)
        assert number_of_entries == 1
        assert number_of_improved_entries == 1

        data_new = modelcreator.MIPData(20, 3.0, 50.0, 2, -1)  # better objective, better bound
        self.tl_low_resolution.add_log_data(data_new, 2.1)  # added 0.1 s after first

        number_of_entries = len(self.tl_low_resolution.log_entries)
        number_of_improved_entries = len(self.tl_low_resolution.improved_entries)
        assert number_of_entries == 1
        assert number_of_improved_entries == 1

        last_entry = self.tl_low_resolution.log_entries[-1]
        assert last_entry.data == data_new
        assert last_entry.data != data_old

        last_improved_entry = self.tl_low_resolution.improved_entries[-1]
        assert last_improved_entry.data == data_new
        assert last_improved_entry.data != data_old

    def test_improved_replacing_entry_is_added_to_improved_entries_if_last_was_not_improved(self):
        data_old = modelcreator.MIPData(10, 1.0, 100.0, 1, -1)
        self.tl_low_resolution.add_log_data(data_old, 2.0)

        data_old = modelcreator.MIPData(10, 1.0, 100.0, 1, -1)
        self.tl_low_resolution.add_log_data(data_old, 20.0)

        number_of_entries = len(self.tl_low_resolution.log_entries)
        number_of_improved_entries = len(self.tl_low_resolution.improved_entries)
        assert number_of_entries == 2
        assert number_of_improved_entries == 1

        data_new = modelcreator.MIPData(20, 3.0, 50.0, 2, -1)  # better objective, better bound
        self.tl_low_resolution.add_log_data(data_new, 24)  # added 0.1 s after first

        number_of_entries = len(self.tl_low_resolution.log_entries)
        number_of_improved_entries = len(self.tl_low_resolution.improved_entries)
        assert number_of_entries == 2
        assert number_of_improved_entries == 2

        last_entry = self.tl_low_resolution.log_entries[-1]
        assert last_entry.data == data_new
        assert last_entry.data != data_old

        last_improved_entry = self.tl_low_resolution.improved_entries[-1]
        assert last_improved_entry.data == data_new
        assert last_improved_entry.data != data_old

    def test_cannot_replace_entries_indefinitely(self):
        data_first_entry = modelcreator.MIPData(10, 1.0, 100.0, 1, -1)
        self.tl_low_resolution.add_log_data(data_first_entry, 1.0)
        data_replace = modelcreator.MIPData(10, 1.1, 100.0, 2, -1)
        self.tl_low_resolution.add_log_data(data_replace, 9.0)
        data_new_entry = modelcreator.MIPData(10, 1.1, 100.0, 2, -1)
        self.tl_low_resolution.add_log_data(data_new_entry, 12.0)

        number_of_entries = len(self.tl_low_resolution.log_entries)
        number_of_improved_entries = len(self.tl_low_resolution.improved_entries)
        assert number_of_entries == 2
        assert number_of_improved_entries == 1

    def test_new_entry_when_a_better_one_is_added_with_delay(self):
        data_old = modelcreator.MIPData(10, 1.0, 100.0, 1, -1)
        self.temp_log.add_log_data(data_old, 1.0)
        data_new = modelcreator.MIPData(10, 1.1, 100.0, 2, -1)
        self.temp_log.add_log_data(data_new, 20.0)

        number_of_entries = len(self.temp_log.log_entries)
        number_of_improved_entries = len(self.temp_log.improved_entries)
        assert number_of_entries == 2
        assert number_of_improved_entries == 2

        last_entry = self.temp_log.log_entries[-1]
        last_improved_entry = self.temp_log.improved_entries[-1]
        assert last_entry.data == data_new
        assert last_improved_entry.data == data_new

    def test_entry_without_improvement_is_not_added_to_improved_entries(self):
        data_old = modelcreator.MIPData(10, 1.0, 100.0, 1, -1)
        self.temp_log.add_log_data(data_old, 1.0)
        data_new = modelcreator.MIPData(10, 1.0, 100.0, 2, -1)
        self.temp_log.add_log_data(data_new, 10)

        number_of_entries = len(self.temp_log.log_entries)
        number_of_improved_entries = len(self.temp_log.improved_entries)
        assert number_of_entries == 2
        assert number_of_improved_entries == 1
        assert self.temp_log.log_entries[-1].data == data_new
        assert self.temp_log.improved_entries[-1].data == data_old

    def test_lpdata_and_mipdata_dont_mix(self):
        mipdata = modelcreator.MIPData(10, 1.0, 100.0, 1, -1)
        self.tl_low_resolution.add_log_data(mipdata, 1.1)

        lpdata = modelcreator.LPData(10, 1.0)
        self.tl_low_resolution.add_log_data(lpdata, 1.2)

        mipdata = modelcreator.MIPData(10, 50.0, 100.0, 1, -1)
        self.tl_low_resolution.add_log_data(mipdata, 1.3)

        number_of_improved_entries = len(self.tl_low_resolution.improved_entries)
        number_of_entries = len(self.tl_low_resolution.log_entries)
        assert number_of_entries == 3
        assert number_of_improved_entries == 2
