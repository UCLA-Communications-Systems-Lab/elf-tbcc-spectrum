from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from code_design import search_output_path, search_output_prefix
from gridsearch import merge_batches, report_results


class GridSearchUtilityTests(unittest.TestCase):
    def test_output_names(self):
        prefix = search_output_prefix("config/k11n30v5.yaml")
        self.assertEqual(prefix, "k11n30v5_gridsearch")
        self.assertEqual(search_output_path("results", prefix).name, "k11n30v5_gridsearch.h5")
        self.assertEqual(search_output_path("results", prefix, 3).name, "k11n30v5_gridsearch_batch3.h5")

    def test_merge_and_report(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            prefix = "k11n30v5_gridsearch"
            for index, (name, value) in enumerate((("code_a", 1e-6), ("code_b", 2e-6))):
                batch_path = search_output_path(output_dir, prefix, index)
                with h5py.File(batch_path, "w") as result_file:
                    group = result_file.create_group(name)
                    group.create_dataset("dsub_pcw", data=value)
                    group.create_dataset("gpu_spectrum", data=np.array([1, 0, 3]))

            merged_path = merge_batches(output_dir, prefix)
            with h5py.File(merged_path, "r") as merged_file:
                self.assertEqual(sorted(merged_file.keys()), ["code_a", "code_b"])

            records = report_results(merged_path)
            self.assertEqual([record["name"] for record in records], ["code_a"])
            self.assertEqual(records[0]["spectrum"].tolist(), [1, 0, 3])

            matching_records = report_results(merged_path, pattern="code")
            self.assertEqual([record["name"] for record in matching_records], ["code_a", "code_b"])
