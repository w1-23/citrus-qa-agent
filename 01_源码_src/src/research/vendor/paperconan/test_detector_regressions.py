"""Trigger and benign-control fixtures for the imported numeric detector families."""
from __future__ import annotations

from ._sheet import Sheet
from .detectors import (
    detect_arithmetic_progression,
    detect_block_value_duplication,
    detect_dispersed_repeats,
    detect_decimal_tail_clustering,
    detect_equal_pairs,
    detect_grim_grimmer,
    detect_identical_after_rounding,
    detect_last_digit,
    detect_relations,
    detect_repeated_decimals,
    detect_row_pair_digit_coupling,
    detect_row_relations,
    detect_within_column_patterns,
)
from ._audit import (
    detect_collisions,
    detect_cross_sheet_column_duplicates,
    detect_local_pair_runs,
    detect_recurring_row_vectors,
    detect_row_pair_shared_fraction,
    detect_scaled_row_reuse,
    detect_short_row_reuse,
    detect_within_row_shared_fraction,
    detect_within_sheet_fraction_reuse,
)


def _kinds(findings):
    return {finding["kind"] for finding in findings}


def test_column_relations_flag_offset_sum_and_small_discrete_differences():
    x = [1.17 + i * 0.83 for i in range(12)]
    sum_sheet = Sheet.from_rows([[a, 20 - a] for a in x])
    assert "sum_constant" in _kinds(detect_relations(sum_sheet, 0, 12, 0, 2, ["a", "b"]))

    diff_sheet = Sheet.from_rows([[a, a + (0.17 if i % 2 else 0.39)] for i, a in enumerate(x)])
    assert "small_diff_set" in _kinds(detect_relations(diff_sheet, 0, 12, 0, 2, ["a", "b"]))

    ratio_sheet = Sheet.from_rows([[a, 3 * a] for a in x])
    assert "constant_ratio" in _kinds(detect_relations(ratio_sheet, 0, 12, 0, 2, ["a", "b"]))

    affine_x = [0.0, 1.1, 2.7, 4.2, 7.3, 9.8, 12.4, 14.1]
    affine = Sheet.from_rows([[a, 2 * a + 3] for a in affine_x])
    assert "exact_linear" in _kinds(detect_relations(affine, 0, 8, 0, 2, ["a", "b"]))

    fractional_x = [100.1234567, 85.8535172, 77.4318674, 91.5512391,
                    62.7391842, 54.2839167, 43.6172953, 38.9284716]
    shifts = [10, 20, 30, 10, 40, 20, 30, 50]
    shared_fraction = Sheet.from_rows([[a, a + shift] for a, shift in zip(fractional_x, shifts)])
    assert "integer_diff_shared_fraction" in _kinds(
        detect_relations(shared_fraction, 0, 8, 0, 2, ["a", "b"]))

    coarse_x = [12.31, 13.42, 14.53, 15.64, 16.75, 17.86, 18.97, 19.28]
    coarse_shifts = [10, 20, 30, 40, 50, 60, 70, 80]
    round_shift = Sheet.from_rows([[a, a + shift]
                                   for a, shift in zip(coarse_x, coarse_shifts)])
    assert "round_shift_shared_fraction" in _kinds(
        detect_relations(round_shift, 0, 8, 0, 2, ["a", "b"]))

    benign = Sheet.from_rows([[a, a + 0.17 + (i % 5) * 0.071] for i, a in enumerate(x)])
    assert not _kinds(detect_relations(benign, 0, 12, 0, 2, ["a", "b"]))


def test_column_partial_offset_and_row_oriented_scaling_have_benign_controls():
    x = [10.123 + i * 0.137 for i in range(24)]
    y = [v + 0.237 if i < 20 else v + (i - 18) * 0.31 for i, v in enumerate(x)]
    sheet = Sheet.from_rows([[a, b] for a, b in zip(x, y)])
    assert "partial_constant_offset" in _kinds(
        detect_relations(sheet, 0, 24, 0, 2, ["a", "b"]))

    source = [float(i * i + 2 * i + 3) for i in range(12)]
    ratio_rows = Sheet.from_rows([["condition-A", *source], ["condition-B", *[2 * v for v in source]]])
    findings = detect_row_relations(ratio_rows, 0, 2, 1, 13, [f"m{i}" for i in range(12)])
    assert "constant_ratio_row" in _kinds(findings)

    unrelated = Sheet.from_rows([["condition-A", *source],
                                 ["condition-B", *[float(i ** 3 + 7) for i in range(12)]]])
    assert not _kinds(detect_row_relations(unrelated, 0, 2, 1, 13,
                                           [f"m{i}" for i in range(12)]))

    baseline = [10.12 + i * 1.25 for i in range(12)]
    shifted = [value + 20 for value in baseline]
    digit_copy = Sheet.from_rows([["A", *baseline], ["B", *shifted]])
    assert "row_pair_digit_coupling" in _kinds(
        detect_row_pair_digit_coupling(digit_copy, 0, 2, 1, 13,
                                       [f"m{i}" for i in range(12)]))


def test_single_column_patterns_and_progressions_have_trigger_and_control():
    values = [11, 12, 14, 15, 16, 18] * 4
    sheet = Sheet.from_rows([[v] for v in values])
    kinds = _kinds(detect_within_column_patterns(sheet, 0, len(values), 0, 1, ["reading"]))
    assert "missing_last_digits" in kinds

    dominant = Sheet.from_rows([[value] for value in [1, 1, 1, 1, 2, 3, 4, 5]])
    assert "within_col_value_duplication" in _kinds(
        detect_within_column_patterns(dominant, 0, 8, 0, 1, ["reading"]))

    rounded_values = [float(i) + 0.5 for i in range(20)]
    rounded_half = Sheet.from_rows([[value] for value in rounded_values])
    assert "rounded_to_half_or_int" in _kinds(
        detect_within_column_patterns(rounded_half, 0, 20, 0, 1, ["reading"]))

    repeated = Sheet.from_rows([[float(f"{i}.{i % 7:02d}42")] for i in range(1, 25)])
    kinds = _kinds(detect_within_column_patterns(repeated, 0, 24, 0, 1, ["reading"]))
    assert "within_col_decimal_repetition" in kinds

    progression = Sheet.from_rows([[float(i * 3 + 0.2)] for i in range(20)])
    assert "arithmetic_progression" in _kinds(
        detect_arithmetic_progression(progression, 0, 20, 0, 1, ["reading"]))
    normal = Sheet.from_rows([[float(i * i + 0.2)] for i in range(20)])
    assert not _kinds(detect_arithmetic_progression(normal, 0, 20, 0, 1, ["reading"]))


def test_dispersed_high_precision_repeats_and_partial_equal_pairs_have_controls():
    values = [100.123456 + i * 0.713927 + i * i * 0.00317 for i in range(120)]
    for offset in range(10):
        values[60 + offset] = values[offset]
    repeated = Sheet.from_rows([[value] for value in values])
    assert "within_col_dispersed_repeats" in _kinds(
        detect_dispersed_repeats(repeated, 0, 120, 0, 1, ["reading"]))

    independent = Sheet.from_rows([[100.123456 + i * 0.713927 + i * i * 0.00317]
                                   for i in range(120)])
    assert not _kinds(detect_dispersed_repeats(independent, 0, 120, 0, 1, ["reading"]))

    first = [float(i * i + 0.123) for i in range(10)]
    second = [value if i < 6 else value + 0.17 for i, value in enumerate(first)]
    partial = Sheet.from_rows([[a, b] for a, b in zip(first, second)])
    assert "many_equal_pairs" in _kinds(detect_equal_pairs(partial, 0, 10, 0, 2, ["a", "b"]))

    unrelated = Sheet.from_rows([[float(i * i + 0.123), float(i * 7.91 + 0.367)]
                                 for i in range(10)])
    assert not _kinds(detect_equal_pairs(unrelated, 0, 10, 0, 2, ["a", "b"]))


def test_block_duplication_and_rounding_signals_are_not_fired_by_clean_controls():
    matrix = [[1000.123456 + r * 13.17293 + c * 0.41371 for c in range(6)] for r in range(6)]
    matrix[4][4] = matrix[0][0]
    matrix[5][5] = matrix[1][1]
    copied = Sheet.from_rows(matrix)
    assert "block_value_duplication" in _kinds(
        detect_block_value_duplication(copied, 0, 6, 0, 6, [f"c{i}" for i in range(6)]))

    clean = Sheet.from_rows([[1000.123456 + r * 19.73129 + c * 0.36781
                              for c in range(6)] for r in range(6)])
    assert not _kinds(detect_block_value_duplication(clean, 0, 6, 0, 6,
                                                     [f"c{i}" for i in range(6)]))

    same_display = [1.111, 1.122, 1.133, 1.144, 1.155] * 4
    rounded = Sheet.from_rows([[v] for v in same_display])
    assert "identical_after_rounding" in _kinds(
        detect_identical_after_rounding(rounded, 0, 20, 0, 1, ["values"]))


def test_fraction_reuse_and_cross_sheet_column_duplicates_have_controls():
    row = ["sample", 20.1234567, 100.1234567, 35.8535172, 85.8535172]
    within_row = Sheet.from_rows([row])
    grid = {("study.xlsx", "source"): within_row}
    assert "within_row_shared_fraction" in _kinds(detect_within_row_shared_fraction(grid))

    rows = [["A", 20.1234567, 20.8535172, 20.4318674, 20.5512391],
            ["B", 95.1234567, 85.8535172, 91.4318674, 97.5512391]]
    row_pair = {("study.xlsx", "source"): Sheet.from_rows(rows)}
    assert "shared_fraction_row_pair" in _kinds(detect_row_pair_shared_fraction(row_pair))

    values = [float(i * 7.123 + i * i * 0.017) for i in range(1, 16)]
    first = Sheet.from_rows([["sample", value] for value in values])
    second = Sheet.from_rows([["sample", value] for value in values])
    reused = {("study.xlsx", "Fig1"): first, ("study.xlsx", "Fig2"): second}
    assert detect_cross_sheet_column_duplicates(reused)

    unrelated = Sheet.from_rows([["sample", float(i * i + 0.3)] for i in range(1, 16)])
    control = {("study.xlsx", "Fig1"): first, ("study.xlsx", "Fig2"): unrelated}
    assert not detect_cross_sheet_column_duplicates(control)


def test_cross_sheet_collisions_and_row_reuse_detectors_have_clean_controls():
    # A partial same-position copy across unrelated figure namespaces.
    first_grid = {
        (r, c): float(r * 10 + c + 0.1234567)
        for r in range(8) for c in range(3)
    }
    copied_grid = {pos: value + 100 for pos, value in first_grid.items()}
    for pos in list(first_grid)[:8]:
        copied_grid[pos] = first_grid[pos]
    collisions = detect_collisions({
        ("study.xlsx", "Fig1"): first_grid,
        ("study.xlsx", "Fig2"): copied_grid,
    })
    assert "cross_sheet_position_identical" in _kinds(collisions)

    clean_grid = {
        pos: value + 0.3219876 for pos, value in copied_grid.items()
    }
    assert not _kinds(detect_collisions({
        ("study.xlsx", "Fig1"): first_grid,
        ("study.xlsx", "Fig2"): clean_grid,
    }))

    # The same high-information vector appears at three row sites across two figures.
    vector = [1.17391, 7.39173, 2.11719, 11.37191, 4.82731, 17.19371]
    fig1 = Sheet.from_rows([
        ["group A", *vector],
        ["independent", 2.13719, 3.82173, 6.17119, 7.91371, 9.28173, 12.19371],
        ["group C", *vector],
    ])
    fig2 = Sheet.from_rows([
        ["group D", *[value + 0.3 for value in vector]],
        ["independent", 1.23179, 4.71391, 5.31721, 8.19371, 10.27193, 15.19271],
        ["group F", *vector],
    ])
    recurring = detect_recurring_row_vectors({
        ("study.xlsx", "Fig1"): fig1,
        ("study.xlsx", "Fig2"): fig2,
    })
    assert "recurring_row_vector" in _kinds(recurring)
    assert not _kinds(detect_recurring_row_vectors({
        ("study.xlsx", "Fig1"): fig1,
        ("study.xlsx", "Fig2"): Sheet.from_rows([
            ["group D", 1.91, 3.17, 5.39, 8.11, 12.71, 18.13],
            ["independent", 2.23179, 4.11391, 7.21721, 9.19371, 11.27193, 16.19271],
            ["group F", 3.41, 4.27, 7.83, 9.31, 13.17, 19.23],
        ]),
    }))

    # Long scalar row reuse is checked across separate row bands / figures.
    base = [13.17293 + i * 0.41371 + i * i * 0.01713 for i in range(14)]
    source = Sheet.from_rows([
        ["treatment-A", *base],
        ["treatment-X", *[value * 1.234 for value in base]],
    ])
    target = Sheet.from_rows([
        ["treatment-B", *[value * 2.345 for value in base]],
        ["treatment-Y", *[value * 3.456 for value in base]],
    ])
    assert "scaled_row_reuse" in _kinds(detect_scaled_row_reuse({
        ("study.xlsx", "Fig1"): source,
        ("study.xlsx", "Fig2"): target,
    }))
    unrelated = Sheet.from_rows([
        ["treatment-B", *[float(i ** 2 + 0.31 * i + 0.17) for i in range(14)]],
        ["treatment-Y", *[float(i ** 3 + 2.41 * i + 0.73) for i in range(14)]],
    ])
    assert not _kinds(detect_scaled_row_reuse({
        ("study.xlsx", "Fig1"): source,
        ("study.xlsx", "Fig2"): unrelated,
    }))


def test_within_table_fraction_and_local_pair_run_have_controls():
    fractions = [
        0.1234567, 0.8535172, 0.4318674, 0.5512391,
        0.7391842, 0.2839167, 0.6172953, 0.9284716,
        0.3817231, 0.7462913, 0.1957381, 0.5641729,
    ]
    reused_rows = [[fractions[i * 4 + j] + i * 5 for j in range(4)] for i in range(3)]
    reused_rows += [[None] * 4]
    reused_rows += [[fractions[i * 4 + j] + (10 + i * 10) for j in range(4)] for i in range(3)]
    reused = {("study.xlsx", "source"): Sheet.from_rows(reused_rows)}
    assert "within_table_fraction_reuse" in _kinds(detect_within_sheet_fraction_reuse(reused))

    unrelated_rows = [[float(i * 2.173 + j * 0.381 + 0.1234567) for j in range(4)]
                      for i in range(3)]
    unrelated_rows += [[None] * 4]
    unrelated_rows += [[float(i * 3.713 + j * 0.297 + 0.9182731) for j in range(4)]
                       for i in range(3)]
    assert not _kinds(detect_within_sheet_fraction_reuse({
        ("study.xlsx", "source"): Sheet.from_rows(unrelated_rows),
    }))

    # A high-information offset holds only in the middle of the two measured columns.
    base = [10.137, 11.291, 14.723, 16.382, 18.159, 19.631, 23.517, 24.893,
            27.281, 29.417, 31.129, 35.703, 38.229, 42.617, 45.319, 49.827]
    paired = [2.173, 6.291, 9.723, 14.382]
    paired += [value + 0.37 for value in base[4:12]]
    paired += [5.319, 8.827, 10.293, 13.871]
    local = Sheet.from_rows([[a, b] for a, b in zip(base, paired)])
    assert "local_offset_run" in _kinds(
        detect_local_pair_runs(local, 0, local.nrows, 0, local.ncols, ["a", "b"]))

    clean = Sheet.from_rows([
        [value, value * 1.17 + ((i * 17) % 11) * 0.23]
        for i, value in enumerate(base)
    ])
    assert not _kinds(detect_local_pair_runs(
        clean, 0, clean.nrows, 0, clean.ncols, ["a", "b"]))

    # Short, high-precision duplicate row across separated blocks.
    short_values = [0.1234567, 0.7139271, 0.3472193, 0.8917351, 0.5628137, 0.2391871]
    short_sheet = Sheet.from_rows([
        ["group A", *short_values],
        ["independent", 0.2183719, 0.8172931, 0.4317291,
         0.9237181, 0.6519237, 0.3918271],
        [None] * 7,
        ["group B", *short_values],
    ])
    assert "identical_row_reuse" in _kinds(detect_short_row_reuse({
        ("study.xlsx", "Fig1"): short_sheet,
    }))
    assert not _kinds(detect_short_row_reuse({
        ("study.xlsx", "Fig1"): Sheet.from_rows([
            ["group A", *short_values],
            ["independent", 0.2183719, 0.8172931, 0.4317291,
             0.9237181, 0.6519237, 0.3918271],
            [None] * 7,
            ["group B", 0.1812397, 0.9328171, 0.4137291,
             0.7619237, 0.2951371, 0.8273917],
        ]),
    }))


def test_decimal_tail_and_digit_distribution_are_observable_but_clean_data_is_not_clustered():
    concentrated = [float(f"{i}.{i:03d}123") for i in range(1, 31)]
    cluster = detect_decimal_tail_clustering(concentrated, "trigger")
    assert cluster is not None
    assert cluster["n"] >= 12

    spread = [float(f"{i}.{(i * 7919) % 999:03d}") for i in range(1, 101)]
    assert detect_decimal_tail_clustering(spread, "control") is None
    digit_report = detect_last_digit(spread, "control")
    assert digit_report is not None and digit_report["n"] >= 40
    repeated = detect_repeated_decimals(concentrated * 3, "trigger")
    assert repeated is not None and repeated["top"]


def test_grim_grimmer_only_applies_to_integer_count_summaries():
    impossible = Sheet.from_rows([[0.4, 3, 0.8]])
    findings = detect_grim_grimmer(impossible, 0, 1, 0, 3, ["mean count", "n", "SD"])
    assert "grim_inconsistent" in _kinds(findings)

    valid = Sheet.from_rows([[1.0, 3, 1.0]])
    assert not _kinds(detect_grim_grimmer(valid, 0, 1, 0, 3, ["mean count", "n", "SD"]))

    continuous = Sheet.from_rows([[0.4, 3, 0.8]])
    assert not _kinds(detect_grim_grimmer(continuous, 0, 1, 0, 3,
                                         ["mean percentage", "n", "SD"]))

    impossible_sd = Sheet.from_rows([[0.5, 2, 0.1]])
    assert "grimmer_inconsistent" in _kinds(
        detect_grim_grimmer(impossible_sd, 0, 1, 0, 3, ["mean count", "n", "SD"]))
