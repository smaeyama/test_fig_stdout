#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pytest
import os
from bzx import bzx, read_GKV_metric_file

@pytest.fixture
def generate_test_data():
    """Runs BZX to generate a test output file"""
    Ntheta_gkv = 1
    nrho = 5
    ntht = 12
    nzeta = 4
    alpha_fix = 0.1
    fname_boozmn = "./reference_data/boozmn_tests.nc"
    fname_wout = "./reference_data/wout_tests.nc"
    output_file = "./metric_boozer.bin.dat"

    # Run BZX to generate the test file
    bzx(Ntheta_gkv, nrho, ntht, nzeta, alpha_fix, fname_boozmn, fname_wout, output_file)

    # Ensure the output file exists
    assert os.path.exists(output_file), f"Test output file {output_file} was not created!"

    return output_file

def test_gkv_metric_comparison(generate_test_data):
    """Compares reference data and generated test data using tuples"""
    reference_file = "./reference_data/metric_boozer.bin.dat"
    test_file = generate_test_data

    # Read reference and test data as tuples
    reference_data = read_GKV_metric_file(reference_file)
    test_data = read_GKV_metric_file(test_file)

    # Compare each element in the tuples
    keys = ("nfp_b", "nss", "ntht", "nzeta", "mnboz_b", "mboz_b", "nboz_b", "Rax", "Bax", "aa", "volume_p", "asym_flg", "alpha_fix",
            "rho", "theta", "zeta", "qq", "shat", "epst", "bb", "rootg_boz", "rootg_boz0", "ggup_boz",
            "dbb_drho", "dbb_dtht", "dbb_dzeta",
            "rr", "zz", "ph", "bbozc", "ixn_b", "ixm_b",
            "bbozs")
    for i, (ref_val, test_val) in enumerate(zip(reference_data, test_data)):
        if isinstance(ref_val, np.ndarray):  # Compare array values
            if np.isnan(ref_val).any() or np.isinf(ref_val).any():
                print(f"Warning: Reference data at index {i} ({keys[i]}) contains NaN or Inf")
            if np.isnan(test_val).any() or np.isinf(test_val).any():
                print(f"Warning: Test data at index {i} ({keys[i]}) contains NaN or Inf")

            assert np.allclose(ref_val, test_val, atol=1e-8, equal_nan=True), f"Mismatch at index {i} ({keys[i]}): Arrays differ!"
        else:  # Compare scalar values
            assert ref_val == test_val, f"Mismatch at index {i} ({keys[i]}): {ref_val} != {test_val}"

