#! /usr/bin/env python

"""
Script to compute isotropic 3D-SHORE ODFs.
See [Ozarslan et al NeuroImage 2013, Fick et al. 2014-2015].

WARNING: need shore_ozarslan.py which is currently not in DIPY master.
"""

from __future__ import division, print_function

import argparse
import logging

import nibabel as nib
import numpy as np

from dipy.reconst.shm import sf_to_sh
from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
from dipy.io.gradients import read_bvals_bvecs

from scilpy.reconst.shore_ozarslan import ShoreOzarslanModel
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist, add_force_b0_arg,
                             add_sh_basis_args)
from scilpy.utils.bvec_bval_tools import check_b0_threshold


def buildArgsParser():

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('input', action='store', metavar='input', type=str,
                   help='Path of the input diffusion volume.')

    p.add_argument('bvals', action='store', metavar='bvals',
                   help='Path of the bvals file, in FSL format.')

    p.add_argument('bvecs', action='store', metavar='bvecs',
                   help='Path of the bvecs file, in FSL format.')

    p.add_argument('--radial_order', action='store', dest='radial_order',
                   metavar='int', default=8, type=int,
                   help='Radial order used for the SHORE fit. (Default: 8)')

    p.add_argument('--radial_moment', action='store', dest='radial_moment',
                   metavar='int', default=2, type=int,
                   help='Radial moment used for the ODF reconstruction. ' +
                        '(Default: 2)')

    p.add_argument('--regul_weighting', action='store', dest='regul_weighting',
                   metavar='float', default=0.2, type=float,
                   help='Laplacian weighting for the regularization. '
                        '0.0 will make the generalized cross-validation ' +
                        '(GCV) kick in. (Default: 0.2)')

    add_sh_basis_args(p)

    p.add_argument('--mask', action='store', dest='mask',
                   metavar='', default=None, type=str,
                   help='Path to a binary mask. Only the data inside the ' +
                        'mask will be used for computations and ' +
                        'reconstruction.')

    p.add_argument('--not_all', action='store_true', dest='not_all',
                   help='If set, only saves the files specified using the ' +
                        'file flags. (Default: False)')

    g = p.add_argument_group(title='File flags')
    g.add_argument('--odf', action='store', dest='odf',
                   metavar='file', default='', type=str,
                   help='Output filename for the ODF spherical harmonics ' +
                   'coefficients.')
    g.add_argument('--rtop', action='store', dest='rtop',
                   metavar='file', default='', type=str,
                   help='Output filename for the Return-To-Origin ' +
                   'Probability (rtop).')
    g.add_argument('--msd', action='store', dest='msd',
                   metavar='file', default='', type=str,
                   help='Output filename for the Mean Squared ' +
                   'Displacement (msd).')
    g.add_argument('--pa', action='store', dest='pa',
                   metavar='file', default='', type=str,
                   help='Output filename for the Propagator Anisotropy (pa).')

    add_force_b0_arg(p)
    add_overwrite_arg(p)

    return p


def main():

    parser = buildArgsParser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if not args.not_all:
        if not args.odf:
            args.odf = 'shore_dodf.nii.gz'
        if not args.rtop:
            args.rtop = 'rtop.nii.gz'
        if not args.msd:
            args.msd = 'msd.nii.gz'
        if not args.pa:
            args.pa = 'pa.nii.gz'

    arglist = [args.odf, args.rtop, args.msd, args.pa]

    if args.not_all and not any(arglist):
        parser.error('When using --not_all, you need to specify at least ' +
                     'one file to output.')

    assert_inputs_exist(parser, [args.input, args.bvals, args.bvecs])
    assert_outputs_exist(parser, args, arglist)

    vol = nib.load(args.input)
    data = vol.get_data()
    affine = vol.get_affine()

    bvals, bvecs = read_bvals_bvecs(args.bvals, args.bvecs)
    check_b0_threshold(args, bvals.min())
    gtab = gradient_table(bvals, bvecs, b0_threshold=bvals.min())

    if args.mask is None:
        mask = None
    else:
        mask = nib.load(args.mask).get_data().astype(np.bool)
        voxels_with_values_mask = data[:, :, :, 0] > 0
        mask = voxels_with_values_mask * mask

    sphere = get_sphere('repulsion100')

    if args.regul_weighting <= 0:
        logging.info('Now computing SHORE ODF of radial order {0}'
                     .format(args.radial_order) +
                     ' and Laplacian generalized cross-validation')

        shore_model = ShoreOzarslanModel(gtab, radial_order=args.radial_order,
                                         laplacian_regularization=True,
                                         laplacian_weighting='GCV')
    else:
        logging.info('Now computing SHORE ODF of radial order {0}'
                     .format(args.radial_order) +
                     ' and Laplacian regularization weight of {0}'
                     .format(args.regul_weighting))

        shore_model = ShoreOzarslanModel(gtab, radial_order=args.radial_order,
                                         laplacian_regularization=True,
                                         laplacian_weighting=args.regul_weighting)

    smfit = shore_model.fit(data, mask)
    odf = smfit.odf(sphere, radial_moment=args.radial_moment)
    odf_sh = sf_to_sh(odf, sphere, sh_order=8, basis_type=args.sh_basis,
                      smooth=0.0)

    rtop = smfit.rtop()
    msd = smfit.msd()
    pa = smfit.propagator_anisotropy()

    if args.odf:
        nib.save(nib.Nifti1Image(odf_sh.astype(np.float32), affine), args.odf)

    if args.rtop:
        nib.save(nib.Nifti1Image(rtop.astype(np.float32), affine), args.rtop)

    if args.msd:
        nib.save(nib.Nifti1Image(msd.astype(np.float32), affine), args.msd)

    if args.pa:
        nib.save(nib.Nifti1Image(pa.astype(np.float32), affine), args.pa)


if __name__ == "__main__":
    main()
