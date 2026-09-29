from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import yaml

from code_design import (
    gen_all_elf_tbcc,
    run_grid_search,
    search_output_path,
    search_output_prefix,
)
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

    def test_generator_major_mask_preserves_order_and_names(self):
        pattern = [[1, 1], [1, 0]]
        candidates = gen_all_elf_tbcc(
            K_elf=2, N_elf=2, m=0, N_tbcc=3, nu=2,
            fixed_elf_poly="1", puncture_pattern=pattern,
        )
        ordered = {tuple(item["tbcc_config"]["gen_polys"]) for item in candidates}
        self.assertIn(("7", "3"), ordered)
        self.assertIn(("3", "7"), ordered)
        self.assertTrue(all(item["tbcc_config"]["puncture_pattern"] == pattern
                            for item in candidates))
        first = search_output_prefix("config/k2n3v2.yaml", puncture_mask=np.array(pattern))
        second = search_output_prefix(
            "config/k2n3v2.yaml", puncture_mask=np.array([[1, 0], [1, 1]])
        )
        self.assertNotEqual(first, second)

    def test_merge_checks_puncture_metadata(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            prefix = "punctured_gridsearch"
            pattern = np.array([[1, 1, 0, 1], [1, 0, 1, 1]], dtype=np.uint8)
            for index in (0, 1):
                with h5py.File(search_output_path(output_dir, prefix, index), "w") as result:
                    result.attrs["puncture_pattern"] = pattern
                    result.attrs["mother_N"] = 8
                    result.attrs["transmitted_N"] = 6
                    group = result.create_group(f"code_{index}")
                    group.create_dataset("dsub_pcw", data=0.1)
                    group.create_dataset("gpu_spectrum", data=np.array([1, 1, 2]))
            merged_path = merge_batches(output_dir, prefix)
            with h5py.File(merged_path, "r") as merged:
                np.testing.assert_array_equal(merged.attrs["puncture_pattern"], pattern)
                self.assertEqual(int(merged.attrs["transmitted_N"]), 6)

            with h5py.File(search_output_path(output_dir, prefix, 1), "a") as result:
                result.attrs["puncture_pattern"] = np.array(
                    [[1, 0, 1, 1], [1, 1, 0, 1]], dtype=np.uint8
                )
            with self.assertRaisesRegex(ValueError, "metadata does not match"):
                merge_batches(output_dir, prefix)

            with h5py.File(search_output_path(output_dir, prefix, 1), "a") as result:
                del result.attrs["puncture_pattern"]
            with self.assertRaisesRegex(ValueError, "incomplete puncture metadata"):
                merge_batches(output_dir, prefix)

    def test_grid_search_rejects_large_k_before_gpu_work(self):
        config = {
            "bch_config": {"K": 64, "N": 64, "M": 0, "polynomial": "1"},
            "tbcc_config": {"K": 64, "N": 128, "V": 1,
                            "gen_polys": ["3", "1"]},
            "target_EbNo_dB": 5,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "large.yaml"
            with config_path.open("w") as handle:
                yaml.safe_dump(config, handle)
            output_dir = Path(temporary_directory) / "results"
            with self.assertRaisesRegex(ValueError, "K < 64"):
                run_grid_search(config_path, output_dir=output_dir)
            self.assertFalse(output_dir.exists())
