"""Unit tests for individual functions in main_fltower.py (ticket 1.4)."""

from fltower.main_fltower import extract_well_key


# ---------------------------------------------------------------------------
# extract_well_key
# ---------------------------------------------------------------------------
class TestExtractWellKey:
    def test_standard_filename(self):
        key, (letter, num) = extract_well_key("Experiment - P1 - Caff_A1.fcs")
        assert key == "A1"
        assert letter == "A"
        assert num == 1

    def test_two_digit_well(self):
        key, (letter, num) = extract_well_key("Sample_H12.fcs")
        assert key == "H12"
        assert letter == "H"
        assert num == 12

    def test_multiple_well_patterns_takes_last(self):
        # B2 appears first, but A3 is the last well pattern
        key, _ = extract_well_key("B2_something_A3.fcs")
        assert key == "A3"

    def test_no_well_pattern(self):
        key, (name, num) = extract_well_key("random_file.fcs")
        assert num == 0  # fallback: no match

    def test_full_path(self):
        key, _ = extract_well_key("/data/experiment/Sample_C5.fcs")
        assert key == "C5"

    def test_no_extension(self):
        key, _ = extract_well_key("Sample_D7")
        assert key == "D7"
