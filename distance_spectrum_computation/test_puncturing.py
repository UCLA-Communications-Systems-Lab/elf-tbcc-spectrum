"""Independent small-code checks for generator-major puncturing."""

import copy
from itertools import product
from pathlib import Path
import unittest

import numpy as np
import yaml

from setup import setup_A_Wbit_D, spectrum_filename
from step import trellisStep_shift


def explicit_spectrum(config):
    """Encode each message bit by bit without using trellis branch weights."""
    bch = config["bch_config"]
    tbcc = config["tbcc_config"]
    stages = bch["K"] + bch["M"]
    generators = tbcc["gen_polys"]
    mask = tbcc.get("puncture_pattern", [[1] for _ in generators])
    period = len(mask[0])
    elf = [int(bit) for bit in reversed(bch["polynomial"])]
    combined = []
    for polynomial in generators:
        value = int(polynomial, 8)
        taps = [(value >> i) & 1 for i in range(value.bit_length())]
        coefficients = [0] * (len(elf) + len(taps) - 1)
        for i, elf_bit in enumerate(elf):
            for j, generator_bit in enumerate(taps):
                coefficients[i + j] ^= elf_bit & generator_bit
        combined.append(coefficients)

    spectrum = [0] * (tbcc["N"] + 1)
    for message in product((0, 1), repeat=bch["K"]):
        inputs = message + (0,) * bch["M"]
        transmitted = []
        for stage in range(stages):
            for generator, coefficients in enumerate(combined):
                bit = sum(
                    tap * inputs[(stage - delay) % stages]
                    for delay, tap in enumerate(coefficients)
                ) % 2
                if mask[generator][stage % period]:
                    transmitted.append(bit)
        spectrum[sum(transmitted)] += 1
    return spectrum


def trellis_spectrum(config):
    starts, weights, destinations, basis, stages, _, widths = setup_A_Wbit_D(config)
    spectrum = np.zeros(config["tbcc_config"]["N"] + 1, dtype=np.uint64)
    for start, initial in zip(basis, starts):
        counts = initial.copy()
        for stage in range(stages):
            phase = stage % len(widths)
            counts = trellisStep_shift(
                counts, weights[phase], destinations, int(widths[phase])
            )
        spectrum += counts[start]
    return spectrum, weights, widths


class PuncturingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).parent / "config" / "k4n6v1.yaml"
        with path.open() as handle:
            cls.fixture = yaml.safe_load(handle)

    def test_four_stage_fixture(self):
        spectrum, weights, widths = trellis_spectrum(self.fixture)
        self.assertEqual(widths.tolist(), [2, 1, 1, 2])
        self.assertEqual(weights.shape, (4, 2, 2))
        self.assertEqual(weights.tolist(), [
            [[0, 2], [1, 1]],
            [[0, 1], [1, 0]],
            [[0, 1], [0, 1]],
            [[0, 2], [1, 1]],
        ])
        self.assertEqual(spectrum.tolist(), [1, 1, 2, 6, 5, 1, 0])
        self.assertEqual(spectrum.tolist(), explicit_spectrum(self.fixture))

    def test_partial_final_period_with_elf(self):
        config = {
            "bch_config": {"K": 2, "N": 3, "M": 1, "polynomial": "11"},
            "tbcc_config": {
                "K": 3, "N": 5, "V": 1,
                "gen_polys": ["3", "1"],
                "puncture_pattern": [[1, 1], [1, 0]],
            },
        }
        spectrum, _, widths = trellis_spectrum(config)
        self.assertEqual(widths.tolist(), [2, 1])
        self.assertEqual(spectrum.tolist(), [1, 0, 0, 2, 1, 0])
        self.assertEqual(spectrum.tolist(), explicit_spectrum(config))

    def test_zero_width_phase_still_counts_paths(self):
        config = {
            "bch_config": {"K": 2, "N": 2, "M": 0, "polynomial": "1"},
            "tbcc_config": {
                "K": 2, "N": 1, "V": 1,
                "gen_polys": ["3", "1"],
                "puncture_pattern": [[1, 0], [0, 0]],
            },
        }
        spectrum, _, widths = trellis_spectrum(config)
        self.assertEqual(widths.tolist(), [1, 0])
        self.assertEqual(spectrum.tolist(), [2, 2])
        self.assertEqual(spectrum.tolist(), explicit_spectrum(config))

    def test_three_generators_and_generator_order(self):
        config = {
            "bch_config": {"K": 3, "N": 3, "M": 0, "polynomial": "1"},
            "tbcc_config": {
                "K": 3, "N": 6, "V": 1,
                "gen_polys": ["3", "1", "2"],
                "puncture_pattern": [[1, 0], [1, 1], [0, 1]],
            },
        }
        spectrum, weights, widths = trellis_spectrum(config)
        self.assertEqual(weights.shape, (2, 2, 2))
        self.assertEqual(widths.tolist(), [2, 2])
        self.assertEqual(spectrum.tolist(), explicit_spectrum(config))

        ordered = {
            "bch_config": {"K": 2, "N": 2, "M": 0, "polynomial": "1"},
            "tbcc_config": {
                "K": 2, "N": 3, "V": 1,
                "gen_polys": ["3", "1"],
                "puncture_pattern": [[1, 1], [1, 0]],
            },
        }
        swapped = copy.deepcopy(ordered)
        swapped["tbcc_config"]["gen_polys"] = ["1", "3"]
        self.assertNotEqual(trellis_spectrum(ordered)[0].tolist(),
                            trellis_spectrum(swapped)[0].tolist())
        self.assertNotEqual(spectrum_filename(ordered), spectrum_filename(swapped))

    def test_omitted_and_all_ones_masks_match(self):
        mother = copy.deepcopy(self.fixture)
        mother["tbcc_config"].pop("puncture_pattern")
        mother["tbcc_config"]["N"] = 8
        explicit = copy.deepcopy(mother)
        explicit["tbcc_config"]["puncture_pattern"] = [[1], [1]]
        for left, right in zip(trellis_spectrum(mother), trellis_spectrum(explicit)):
            np.testing.assert_array_equal(left, right)
        self.assertEqual(spectrum_filename(mother), "k4n8v1_dist_spectrum.npy")

    def test_mask_validation(self):
        bad_patterns = (
            [[1, 0]],
            [[1, 0], [1]],
            [[1, 2], [1, 1]],
            [[True, 1], [1, 1]],
            [[], []],
            [[1] * 5, [1] * 5],
            [[0], [0]],
        )
        for pattern in bad_patterns:
            with self.subTest(pattern=pattern):
                config = copy.deepcopy(self.fixture)
                config["tbcc_config"]["puncture_pattern"] = pattern
                with self.assertRaises(ValueError):
                    setup_A_Wbit_D(config)

        wrong_length = copy.deepcopy(self.fixture)
        wrong_length["tbcc_config"]["N"] = 8
        with self.assertRaisesRegex(ValueError, "transmitted punctured length"):
            setup_A_Wbit_D(wrong_length)

        wrong_elf_length = copy.deepcopy(self.fixture)
        wrong_elf_length["bch_config"]["N"] = 5
        wrong_elf_length["tbcc_config"]["K"] = 5
        with self.assertRaisesRegex(ValueError, "bch_config.N"):
            setup_A_Wbit_D(wrong_elf_length)


if __name__ == "__main__":
    unittest.main()
