#!/usr/bin/env python
# coding: utf-8

import vmecpp
vmecin = vmecpp.VmecInput.from_file("./input.tests")
vmecout = vmecpp.run(vmecin)
vmecout.wout.save("wout_tests.nc")

import booz_xform as bx
b = bx.Booz_xform()
b.read_wout("wout_tests.nc")
b.run()
b.write_boozmn("boozmn_tests.nc")

from bzx import bzx
Ntheta_gkv = 1   # N_tht value in GKV
nrho = 5         # radial grid number in [0 <= rho <= 1]
ntht = 12        # poloidal grid number in [-N_theta*pi < theta < N_theta*pi]
nzeta = 4        # nzeta==0 for output GKV field-aligned coordinates
alpha_fix = 0.1  # field-line label: alpha = zeta - q*theta NOT USED in 3d case (nzeta > 1)
fname_wout="wout_tests.nc"
fname_boozmn="boozmn_tests.nc"
bzx(Ntheta_gkv, nrho, ntht, nzeta, alpha_fix, fname_boozmn, fname_wout, output_file="./metric_boozer.bin.dat")

