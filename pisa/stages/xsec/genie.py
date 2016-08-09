"""
The purpose of this stage is the conversion of the neutrino flux at the
detector into an event count.

This service in particular reads in from a GENIE cross-section spline ROOT file
in order to obtain the relavant cross-sections.
"""
from operator import add

import numpy as np
import pint; ureg = pint.UnitRegistry()

import ROOT

from pisa.core.stage import Stage
from pisa.core.map import Map
from pisa.core.transform import BinnedTensorTransform, TransformSet
from pisa.core.binning import OneDimBinning, MultiDimBinning
from pisa.utils.flavInt import flavintGroupsFromString, ALL_NUINT_TYPES
from pisa.utils.flavInt import NuFlavInt, NuFlavIntGroup, FlavIntData
from pisa.utils.spline import Spline, CombinedSpline
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging
from pisa.utils.profiler import profile
from pisa.resources.resources import find_resource


class genie(Stage):
    """Aeff service to tranform from flux maps to event rate maps.

    Obtains in cross-section values from a ROOT file containing the GENIE
    cross-section splines. Along with detector fiducial volume and livetime,
    tranforms the flux maps into an event rate.

    Parameters
    ----------
    params: ParamSet of sequence with which to instantiate a ParamSet
        Parameters which set everything besides the binning

        Parameters required by this service are
            * xsec_file : filepath
                Filepath to ROOT file containing GENIE cross-section splines.

            * livetime : ureg.Quantity
                Desired lifetime.

            * ice_p : ureg.Quantity
                Density of detector ice.

            * fid_vol : ureg.Quantity
                Desired fiducial volume.

            * mr_h20 : ureg.Quantity
                Relative atmoic mass of water (ice).

            * x_energy_scale : float
                A proxy systematic designed to account for any uncertainty on
                the overall energy measurements. i.e. an energy scale of 0.9
                says all energies are systematically reconstructed at 90% of
                the truth (on average). This systematic works by evaluating the
                cross-section splines at energy E*energy_scale and reading it
                in to energy bin E.

    input_binning : MultiDimBinning or convertible thereto
        Input binning is in true variables, with names prefixed by "true_".
        Each must match a corresponding dimension in `output_binning`.

    output_binning : MultiDimBinning or convertible thereto
        Output binning is in reconstructed variables, with names (traditionally
        in PISA but not necessarily) prefixed by "reco_". Each must match a
        corresponding dimension in `input_binning`.

    error_method : None, bool, or string
        If None, False, or empty string, the stage does not compute errors for
        the transforms and does not apply any (additional) error to produce its
        outputs. (If the inputs already have errors, these are propagated.)

    debug_mode : None, bool, or string
        If None, False, or empty string, the stage runs normally.
        Otherwise, the stage runs in debug mode. This disables caching (forcing
        recomputation of any nominal transforms, transforms, and outputs).

    disk_cache : None, str, or DiskCache
        If None, no disk cache is available.
        If str, represents a path with which to instantiate a utils.DiskCache
        object. Must be concurrent-access-safe (across threads and processes).

    transforms_cache_depth
    outputs_cache_depth : int >= 0

    Input Names
    ----------
    The `inputs` container must include objects with `name` attributes:
        * 'nue'
        * 'nuebar'
        * 'numu'
        * 'numubar'

    Output Names
    ----------
    The `outputs` container generated by this service will be objects with the
    following `name` attribute:
        * 'nue_cc'
        * 'nue_nc'
        * 'nuebar_cc'
        * 'nuebar_nc'
        * 'numu_cc'
        * 'numu_nc'
        * 'numubar_cc'
        * 'numubar_nc'

    """
    def __init__(self, params, transform_groups, input_binning, output_binning,
                 error_method=None, debug_mode=None, disk_cache=None,
                 transforms_cache_depth=20, outputs_cache_depth=20):
        self.xsec_hash = None
        """Hash of GENIE spline file"""

        expected_params = (
            'xsec_file', 'livetime', 'ice_p', 'fid_vol', 'mr_h20',
            'x_energy_scale'
        )

        input_names = (
            'nue', 'nuebar', 'numu', 'numubar', 'nutau', 'nutaubar'
        )

        all_names = (
            'nue_cc', 'nuebar_cc', 'numu_cc', 'numubar_cc', 'nutau_cc', 'nutaubar_cc',
            'nue_nc', 'nuebar_nc', 'numu_nc', 'numubar_nc', 'nutau_nc', 'nutaubar_nc',
        )
        if transform_groups is None:
            output_names = all_names
        else:
            transform_groups = flavintGroupsFromString(transform_groups)
            output_names = []
            for grp in transform_groups:
                flavints = [str(g) for g in grp.flavints()]
                if set(flavints).intersection(all_names) \
                   and str(grp) not in output_names:
                    output_names.append(str(grp))
        self.transform_groups = [NuFlavIntGroup(flavint)
                                 for flavint in output_names]

        super(self.__class__, self).__init__(
            use_transforms=True,
            stage_name='xsec',
            service_name='genie',
            params=params,
            expected_params=expected_params,
            input_names=input_names,
            output_names=output_names,
            error_method=error_method,
            disk_cache=disk_cache,
            outputs_cache_depth=outputs_cache_depth,
            transforms_cache_depth=transforms_cache_depth,
            input_binning=input_binning,
            output_binning=output_binning,
            debug_mode=debug_mode
        )

        self.include_attrs_for_hashes('transform_groups')

    @profile
    def _compute_nominal_transforms(self):
        """Compute cross-section transforms."""
        logging.info('Updating xsec.genie cross-section histograms...')

        self.load_xsec_splines()
        livetime = self._ev_param(self.params['livetime'].value)
        ice_p    = self._ev_param(self.params['ice_p'].value)
        fid_vol  = self._ev_param(self.params['fid_vol'].value)
        mr_h20   = self._ev_param(self.params['mr_h20'].value)
        x_energy_scale = self.params['x_energy_scale'].value

        input_binning = self.input_binning
        assert input_binning == self.output_binning

        ebins = input_binning.true_energy
        for idx, name in enumerate(input_binning.names):
            if 'true_energy' in name:
                e_idx = idx

        xsec_transforms = {}
        for flav in self.input_names:
            for Int in ALL_NUINT_TYPES:
                flavint = flav + '_' + str(Int)
                logging.debug('Obtaining cross-sections for '
                              '{0}'.format(flavint))
                xsec_map = self.xsec.get_map(
                    flavint, MultiDimBinning(ebins),
                    x_energy_scale=x_energy_scale
                )

                def x(idx):
                    if idx == e_idx: return xsec_map.hist
                    else: return range(input_binning.shape[idx])
                num_dims = input_binning.num_dims
                xsec_trns = np.meshgrid(*map(x, range(num_dims)),
                                        indexing='ij')[e_idx]
                xsec_trns *= livetime * fid_vol * \
                    (ice_p / mr_h20) * (6.022140857e+23 / ureg.mol)
                xsec_transforms[NuFlavInt(flavint)] = xsec_trns

        nominal_transforms = []
        for flav_int_group in self.transform_groups:
            flav_names = [str(flav) for flav in flav_int_group.flavs()]
            for input_name in self.input_names:
                if input_name not in flav_names:
                    continue

                xform_array = []
                for flav_int in flav_int_group.flavints():
                    if flav_int in xsec_transforms:
                        xform_array.append(xsec_transforms[flav_int])
                xform_array = reduce(add, xform_array)

                xform = BinnedTensorTransform(
                    input_names=input_name,
                    output_name=str(flav_int_group),
                    input_binning=input_binning,
                    output_binning=self.output_binning,
                    xform_array=xform_array
                )
                nominal_transforms.append(xform)

        return TransformSet(transforms=nominal_transforms)

    def load_xsec_splines(self):
        """Load the cross-sections splines from the ROOT file."""
        xsec_file = self.params['xsec_file'].value
        this_hash = hash_obj(xsec_file)
        if this_hash == self.xsec_hash:
            self.xsec.reset()
            return

        logging.info('Extracting cross-section spline from file: '
                     '{0}'.format(xsec_file))
        self.xsec = self.get_combined_xsec(xsec_file, ver='v2.10.0')
        self.xsec_hash = this_hash

    @staticmethod
    def get_combined_xsec(fpath, ver=None):
        """Load the cross-section values from a ROOT file and instantiate a
        CombinedSpline object."""
        fpath = find_resource(fpath)
        logging.info('Loading GENIE ROOT cross-section file {0}'.format(fpath))

        flavs = (
            'nu_e', 'nu_mu', 'nu_tau',
            'nu_e_bar', 'nu_mu_bar', 'nu_tau_bar'
        )
        """Name of neutrino flavours in the ROOT file."""

        rfile = ROOT.TFile.Open(fpath, 'read')
        xsec_splines = FlavIntData()
        for flav in flavs:
            for Int in ALL_NUINT_TYPES:
                xsec_splines[flav, Int] = {}
                for part in ('O16', 'H1'):
                    str_repr = flav+'_'+part+'/'+'tot_'+str(Int)
                    xsec_splines[flav + str(Int)][part] = \
                        ROOT.gDirectory.Get(str_repr)
        rfile.Close()

        def eval_spl(spline, binning, out_units=ureg.m**2, x_energy_scale=1.,
                     **kwargs):
            num_dims = 1
            init_names = ['true_energy']
            init_units = [ureg.GeV]

            if set(binning.names) != set(init_names):
                raise ValueError('Input binning names {0} does not match '
                                 'instantiation binning names '
                                 '{1}'.format(binning.names, init_names))

            if set(map(str, binning.units)) != set(map(str, init_units)):
                for name in init_names:
                    binning[name].to(init_units)

            bin_centers = [x.m for x in binning.weighted_centers][0]

            nu_O16, nu_H1 = [], []
            for e_val in bin_centers:
                nu_O16.append(spline['O16'].Eval(e_val))
                nu_H1.append(spline['H1'].Eval(e_val))

            nu_O16, nu_H1 = map(np.array, (nu_O16, nu_H1))
            nu_xsec = ((0.8879*nu_O16) + (0.1121*nu_H1)) * 1E-38 * ureg.cm**2

            nu_xsec_hist = nu_xsec.to(out_units).magnitude
            return Map(hist=nu_xsec_hist, binning=binning, **kwargs)


        def validate_spl(binning):
            if np.all(binning.true_energy.midpoints.m > 1E3):
                raise ValueError('Energy value {0} out of range in array '
                                 '{0}'.format(binning.true_energy))

        inXSec = []
        for flav in flavs:
            for Int in ALL_NUINT_TYPES:
                flavint = NuFlavInt(flav+str(Int))
                xsec = Spline(
                    name=str(flavint), spline=xsec_splines[flavint],
                    eval_spl=eval_spl, validate_spl=validate_spl
                )
                inXSec.append(xsec)

        return CombinedSpline(inXSec, interactions=True, ver=ver)

    @profile
    def _compute_transforms(self):
        # TODO(shivesh): this
        return self.nominal_transforms

    @staticmethod
    def _ev_param(parameter):
        if isinstance(parameter, basestring):
            return eval(parameter)
        else: return parameter

    def validate_params(self, params):
        assert isinstance(params['xsec_file'].value, basestring)
        assert isinstance(self._ev_param(params['livetime'].value), ureg.Quantity)
        assert isinstance(self._ev_param(params['ice_p'].value), ureg.Quantity)
        assert isinstance(self._ev_param(params['fid_vol'].value), ureg.Quantity)
        assert isinstance(self._ev_param(params['mr_h20'].value), ureg.Quantity)
        assert (params['x_energy_scale'].value > 1E-5)


def test_standard_plots(xsec_file, outdir='./'):
    from CFXToy.utils.plotter import plotter
    xsec = genie.get_combined_xsec(xsec_file)

    e_bins = MultiDimBinning(OneDimBinning(name='true_energy', tex=r'E$_\nu$',
                                           num_bins=150, domain=(1E-1, 1E3)*ureg.GeV,
                                           is_log=True))
    xsec.compute_maps(e_bins)

    logging.info('Making plots for genie xsec_maps')
    plot_obj = plotter(outdir=outdir, stamp='Cross-Section', fmt='png',
                       log=True, size=(12, 8),
                       label=r'Cross-Section ($m^{2}$)')
    maps = xsec.return_mapset()
    plot_obj.plot_CFX_xsec(maps, ylim=(1E-43, 1E-37))


def test_per_e_plot(xsec_file, outdir='./'):
    from CFXToy.utils.plotter import plotter
    xsec = genie.get_combined_xsec(xsec_file)

    e_bins = MultiDimBinning(OneDimBinning(name='true_energy', tex=r'E$_\nu$',
                                           num_bins=200, domain=(1E-1, 1E3)*ureg.GeV,
                                           is_log=True))
    xsec.compute_maps(e_bins)
    xsec.scale_maps(1./e_bins.true_energy.bin_widths)

    logging.info('Making plots for genie xsec_maps per energy')
    plot_obj = plotter(outdir=outdir, stamp='Cross-Section / Energy',
                       fmt='png', log=False, size=(12, 8),
                       label=r'Cross-Section / Energy ($m^{2} GeV^{-1}$)')
    maps = xsec.return_mapset()
    plot_obj.plot_CFX_xsec(maps, ylim=(3.5E-41, 3E-40))

if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    from pisa.utils.log import set_verbosity
    set_verbosity(3)

    parser = ArgumentParser(description='Test genie xsec service',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-x', '--xsec_file', type=str, required=True,
                        help='Input GENIE ROOT cross-section file')
    parser.add_argument('-o', '--outdir', type=str, default='./',
                        help='Output directory for plots')
    args = parser.parse_args()

    # TODO(shivesh): tests
    # test_XSec(args.xsec_file)
    test_standard_plots(args.xsec_file, args.outdir+'/standard/')
    test_per_e_plot(args.xsec_file, args.outdir+'/per_e/')
