import copy
from pathlib import Path
import unittest

import numpy as np
import yaml

from code_design import gen_all_elf_tbcc, generator_key
from setup import setup_A_Wbit_D
from step import trellisStep_shift


CONFIG_DIR = Path(__file__).parent / "config"


def cpu_spectrum(config):
    As, weights, destinations, basis, stages, max_shift = setup_A_Wbit_D(config)
    spectrum = np.zeros(1 + max_shift * stages, dtype=np.uint64)
    for stream, initial in enumerate(As):
        result = initial.copy()
        for _ in range(stages):
            result = trellisStep_shift(result, weights, destinations, max_shift)
        spectrum[: result.shape[1]] += result[basis[stream]]
    return spectrum, weights


class RateSupportTests(unittest.TestCase):
    def load_config(self, name):
        with (CONFIG_DIR / name).open() as handle:
            return yaml.safe_load(handle)

    def test_fixed_rate_one_third_cpu_spectrum(self):
        config = {
            "bch_config": {"K": 11, "N": 15, "M": 4, "polynomial": "10011"},
            "tbcc_config": {
                "K": 15,
                "N": 45,
                "V": 3,
                "gen_polys": ["13", "15", "17"],
            },
        }
        spectrum, weights = cpu_spectrum(config)
        self.assertEqual(weights.shape[1], 2)  # one binary input bit per stage
        self.assertLessEqual(int(weights.max()), 3)
        self.assertEqual(len(spectrum), 46)
        self.assertEqual(int(spectrum.sum()), 2 ** 11)

    def test_migrated_rate_one_half_cpu_spectrum(self):
        config = self.load_config("k11n22v3.yaml")
        spectrum, weights = cpu_spectrum(config)
        self.assertLessEqual(int(weights.max()), 2)
        self.assertEqual(len(spectrum), 23)
        self.assertEqual(int(spectrum.sum()), 2 ** 11)

    def test_configuration_validation(self):
        config = {
            "bch_config": {"K": 11, "N": 15, "M": 4, "polynomial": "10011"},
            "tbcc_config": {
                "K": 15,
                "N": 45,
                "V": 3,
                "gen_polys": ["13", "15", "17"],
            },
        }
        invalid = copy.deepcopy(config)
        invalid["tbcc_config"]["N"] = 44
        with self.assertRaisesRegex(ValueError, "len\\(gen_polys\\)"):
            setup_A_Wbit_D(invalid)

    def test_rate_one_third_search_uses_unique_triplets(self):
        configs = gen_all_elf_tbcc(
            K_elf=1,
            N_elf=3,
            m=0,
            N_tbcc=9,
            nu=3,
        )
        keys = {
            generator_key("1", item["tbcc_config"]["gen_polys"], 0, 3)
            for item in configs
        }
        self.assertTrue(configs)
        self.assertEqual(len(keys), len(configs))
        self.assertTrue(all(len(item["tbcc_config"]["gen_polys"]) == 3 for item in configs))


if __name__ == "__main__":
    unittest.main()
