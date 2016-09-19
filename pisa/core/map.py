# author : J.L. Lanfranchi
#          jll1062+pisa@phys.psu.edu
#
# date   : March 25, 2016

"""
Map class to contain 2D histogram, error, and metadata about the contents.
MapSet class to contain a set of maps.

Also provide basic mathematical operations that user applies directly to the
containers but that get passed down to operate on the contained data.
"""


from __future__ import division

from collections import OrderedDict, Mapping, Sequence
from copy import deepcopy, copy
from fnmatch import fnmatch
from functools import wraps
from itertools import izip
from operator import add, getitem, setitem
import re

import numpy as np
import scipy.stats as stats
import uncertainties
from uncertainties import ufloat
from uncertainties import unumpy as unp

from pisa.core.binning import OneDimBinning, MultiDimBinning
from pisa.utils.comparisons import isbarenumeric, normQuant, recursiveEquality
from pisa.utils.hash import hash_obj
from pisa.utils import jsons
from pisa.utils.log import logging, set_verbosity
from pisa.utils.random_numbers import get_random_state
from pisa.utils.stats import chi2, llh, conv_llh, mod_chi2
from pisa.utils.profiler import line_profile, profile


HASH_SIGFIGS = 12
VALID_METRICS = ('chi2', 'llh', 'conv_llh', 'mod_chi2')
METRICS_TO_MAXIMIZE = ['llh', 'conv_llh']


# TODO: CUDA and numba implementations of rebin if these libs are available

# TODO: move these functions to a generic utils.py module?

def type_error(value):
    raise TypeError('Type of argument not supported: "%s"' % type(value))


def strip_outer_parens(value):
    value = value.strip()
    m = re.match(r'^\{\((.*)\)\}$', value)
    if m is not None:
        value = m.groups()[0]
    m = re.match(r'^\((.*)\)$', value)
    if m is not None:
        value = m.groups()[0]
    return value


def sanitize_name(name):
    """Make a name a valid Python identifier.

    From Triptych at http://stackoverflow.com/questions/3303312
    """
    # Remove invalid characters
    name = re.sub('[^0-9a-zA-Z_]', '', name)
    # Remove leading characters until we find a letter or underscore
    name = re.sub('^[^a-zA-Z_]+', '', name)
    return name


# TODO: put uncertainties in
def rebin(hist0, orig_binning, new_binning):
    if set(new_binning.basenames) != set(orig_binning.basenames):
        raise ValueError(
            "`new_binning` dimensions' basenames %s do not have 1:1"
            " correspondence (modulo pre/suffixes) to original binning"
            " dimensions' basenames %s"
            %(new_binning.basenames, self.binning.basenames)
        )

    if orig_binning.edges_hash == new_binning.edges_hash:
        return hist0

    orig_dim_indices = []
    new_dim_indices = []
    for new_dim_idx, new_dim in enumerate(new_binning):
        orig_dim_idx = orig_binning.index(new_dim.basename, use_basenames=True)

        new_dim_indices.append(new_dim_idx)
        orig_dim_indices.append(orig_dim_idx)

        orig_dim = orig_binning.dimensions[orig_dim_idx]

        orig_edges = normQuant(orig_dim.bin_edges, sigfigs=HASH_SIGFIGS)
        new_edges = normQuant(new_dim.bin_edges, sigfigs=HASH_SIGFIGS)
        if not np.all(new_edges == orig_edges):
            orig_edge_idx = np.array([np.where(orig_edges == n)
                                      for n in new_edges]).ravel()
            hist0 = np.add.reduceat(hist0, orig_edge_idx[:-1],
                                    axis=orig_dim_idx)

    new_hist = np.moveaxis(hist0, source=orig_dim_indices,
                           destination=new_dim_indices)

    return new_hist


# TODO: implement strategies for decreasing dimensionality (i.e.
# projecting map onto subset of dimensions in the original map)

# TODO: Should all calls to np.<...> be replaced with unp.<...> as is done for
# unp.sqrt below?

class Map(object):
    """Class to contain a multi-dimensional histogram, error, and metadata
    about the histogram. Also provides basic mathematical operations for the
    contained data. See Examples below for how to use a Map object.


    Parameters
    ----------
    name : string
        Name for the map. Used to identify the map.

    hist : numpy.ndarray (incl. obj array from uncertainties.unumpy.uarray)
        The "data" (counts, etc.) in the map.  The shape of `hist` must be
        compatible with the `binning` specified.

    binning : MultiDimBinning
        Describes the binning of the Map.

    error_hist : numpy ndarray
        Must be same shape as `hist`. If specified, sets the error standard
        deviations for the contained `hist`, replacing any stddev information
        that might be contained in the passed `hist` arg.

    hash : None, or immutable object (typically an integer)
        Hash value to attach to the map.

    tex : None or string
        TeX string that can be used for e.g. plotting.

    full_comparison : bool
        Whether to perform full (recursive) comparisons when testing the
        equality of this map with another. See `__eq__` method.


    Examples
    --------
    >>> import pint; ureg = pint.UnitRegistry()
    >>> from pisa.core.binning import MultiDimBinning
    >>> binning = MultiDimBinning([dict(name='energy', is_log=True, num_bins=4,
    ...                                 domain=[1,80]*ureg.GeV),
    ...                            dict(name='coszen', is_lin=True, num_bins=5,
    ...                                 domain=[-1,0])])
    >>> m0 = Map(name='x', binning=binning, hist=np.zeros(binning.shape))
    >>> m0
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    >>> m0.binning
    energy: 4 logarithmically-uniform bins spanning [1.0, 80.0] GeV
    coszen: 5 equally-sized bins spanning [-1.0, 0.0]
    >>> m0.hist[0:4, 0] = 1
    >>> m0
    array([[ 1.,  0.,  0.,  0.,  0.],
           [ 1.,  0.,  0.,  0.,  0.],
           [ 1.,  0.,  0.,  0.,  0.],
           [ 1.,  0.,  0.,  0.,  0.]])
    >>> m1 = m0[0:3, 0:2]
    >>> m1.binning
    energy: 3 logarithmically-uniform bins spanning [1.0, 26.7496121991]
    coszen: 2 equally-sized bins spanning [-1.0, -0.6]
    >>> m1
    array([[ 1.,  0.],
           [ 1.,  0.],
           [ 1.,  0.]])
    >>> for bin in m1.iterbins():
    ...     print '({0:~.2f}, {1:~.2f}): {2:0.1f}'.format(
    ...             bin.binning.energy.midpoints[0],
    ...             bin.binning.coszen.midpoints[0],
    ...             bin.hist[0,0])
    (2.00 GeV, -0.90 ): 1.0
    (2.00 GeV, -0.70 ): 0.0
    (5.97 GeV, -0.90 ): 1.0
    (5.97 GeV, -0.70 ): 0.0
    (17.85 GeV, -0.90 ): 1.0
    (17.85 GeV, -0.70 ): 0.0

    """
    _slots = ('name', 'hist', 'binning', 'hash', '_hash', 'tex',
               'full_comparison', 'parent_indexer')
    _state_attrs = ('name', 'hist', 'binning', 'hash', 'tex',
                    'full_comparison')

    def __init__(self, name, hist, binning, error_hist=None, hash=None,
                 tex=None, full_comparison=True):
        # Set Read/write attributes via their defined setters
        super(Map, self).__setattr__('_name', name)
        # TeX dict for some common map names
        tex_dict = {'nue':r'\nu_e',
                    'numu':r'\nu_{\mu}',
                    'nutau':r'\nu_{\tau}',
                    'nuebar':r'\bar{\nu}_e',
                    'numubar':r'\bar{\nu}_{\mu}',
                    'nutaubar':r'\bar{\nu}_{\tau}',
                    'nue_cc':r'\nu_e\ CC',
                    'numu_cc':r'\nu_{\mu}\ CC',
                    'nutau_cc':r'\nu_{\tau}\ CC',
                    'nue_nc':r'\nu_e\ NC',
                    'numu_nc':r'\nu_{\mu}\ NC',
                    'nutau_nc':r'\nu_{\tau}\ NC',
                    'nuebar_cc':r'\bar{\nu}_e\ CC',
                    'numubar_cc':r'\bar{\nu}_{\mu}\ CC',
                    'nutaubar_cc':r'\bar{\nu}_{\tau}\ CC',
                    'nuebar_nc':r'\bar{\nu}_e\ NC',
                    'numubar_nc':r'\bar{\nu}_{\mu}\ NC',
                    'nutaubar_nc':r'\bar{\nu}_{\tau}\ NC'}
        if tex is None:
            if tex_dict.has_key(name):
                tex = tex_dict[name]
            else:
                tex = r'\rm{%s}' % name
        super(Map, self).__setattr__('_tex', tex)
        super(Map, self).__setattr__('_hash', hash)
        super(Map, self).__setattr__('_full_comparison', full_comparison)

        if not isinstance(binning, MultiDimBinning):
            if isinstance(binning, Sequence):
                binning = MultiDimBinning(dimensions=binning)
            elif isinstance(binning, Mapping):
                binning = MultiDimBinning(**binning)
            else:
                raise ValueError('Do not know what to do with `binning`=%s of'
                                 ' type %s' %(binning, type(binning)))
        self.parent_indexer = None

        # Do the work here to set read-only attributes
        super(Map, self).__setattr__('_binning', binning)
        binning.assert_array_fits(hist)
        super(Map, self).__setattr__('_hist', hist)
        if error_hist is not None:
            self.set_errors(error_hist)

    def __repr__(self):
        argstrs = [('%s=%s' %item) for item in self._serializable_state]
        return '%s(%s)' %(self.__class__.__name__, ', '.join(argstrs))

    def set_poisson_errors(self):
        """Approximate poisson errors using sqrt(n)."""
        super(Map, self).__setattr__('_hist', unp.uarray(self._hist,
                                                         np.sqrt(self._hist)))

    def set_errors(self, error_hist):
        """Manually define the error with an array the same shape as the
        contained histogram. Can also remove errors by passing None.

        Parameters
        ----------
        error_hist : None or ndarray (same shape as hist)
            Standard deviations to apply to `self.hist`.
            If None is passed, any errors present are removed, making
            `self.hist` a bare numpy array.

        """
        if error_hist is None:
            super(Map, self).__setattr__('_hist',
                                         unp.nominal_values(self._hist))
            return
        self.assert_compat(error_hist)
        super(Map, self).__setattr__('_hist', unp.uarray(self._hist,
                                                         error_hist))

    def new_obj(original_function):
        """Decorator to deepcopy unaltered states into new object."""
        @wraps(original_function)
        def new_function(self, *args, **kwargs):
            new_state = OrderedDict()
            state_updates = original_function(self, *args, **kwargs)
            for slot in self._state_attrs:
                if state_updates is not None and state_updates.has_key(slot):
                    new_state[slot] = state_updates[slot]
                else:
                    new_state[slot] = deepcopy(self.__getattr__(slot))
            return Map(**new_state)
        return new_function

    @new_obj
    def reorder_dimensions(self, order):
        new_binning = self.binning.reorder_dimensions(order)
        orig_order = range(len(self.binning))
        new_order = [self.binning.index(b, use_basenames=True)
                     for b in new_binning]
        # TODO: should this be a deepcopy rather than a simple veiw of the
        # original hist (the result of np.moveaxis)?
        new_hist = np.moveaxis(self.hist, source=new_order,
                               destination=orig_order)
        return {'hist': new_hist, 'binning': new_binning}

    @new_obj
    def rebin(self, new_binning):
        """Rebin the map with bin edge lodations and names according to those
        specified in `new_binning`.

        Parameters
        ----------
        new_binning : MultiDimBinning
            Dimensions specified in `new_binning` must match (modulo
            pre/suffixes) the current dimensions.

        Returns
        -------
        Map binned according to `new_binning`.

        """
        # TODO: put uncertainties in
        new_hist = rebin(self.hist, self.binning, new_binning)
        return {'hist': new_hist, 'binning': new_binning}

    def downsample(self, *args, **kwargs):
        """Downsample by integer factors.

        See pisa.utils.binning.downsample for args/kwargs details.

        """
        new_binning = self.binning.downsample(*args, **kwargs)
        return self.rebin(new_binning)

    @new_obj
    def fluctuate(self, method, random_state=None, jumpahead=0):
        orig = method
        method = str(method).lower()
        if method == 'poisson':
            random_state = get_random_state(random_state, jumpahead=jumpahead)
            with np.errstate(invalid='ignore'):
                orig_hist = unp.nominal_values(self.hist)
                nan_at = np.isnan(orig_hist)
                valid_mask = ~nan_at

                hist_vals = np.empty_like(orig_hist, dtype=np.float64)
                hist_vals[valid_mask] = stats.poisson.rvs(
                    orig_hist[valid_mask],
                    random_state=random_state
                )
                hist_vals[nan_at] = np.nan

                error_vals = np.empty_like(orig_hist, dtype=np.float64)
                error_vals[valid_mask] = np.sqrt(orig_hist[valid_mask])
                error_vals[nan_at] = np.nan
            return {'hist': unp.uarray(hist_vals, error_vals)}

        elif method in ['', 'none', 'false']:
            return {}

        else:
            raise Exception('fluctuation method %s not implemented' %orig)

    @property
    def shape(self):
        return self.hist.shape

    @property
    def _serializable_state(self):
        state = OrderedDict()
        state['name'] = self.name
        state['hist'] = unp.nominal_values(self.hist)
        state['binning'] = self.binning._serializable_state
        stddevs = unp.std_devs(self.hist)
        stddevs = None if np.all(stddevs == 0) else stddevs
        state['error_hist'] = stddevs
        state['hash'] = self.hash
        state['tex'] = self.tex
        state['full_comparison'] = self.full_comparison
        return state

    @property
    def _hashable_state(self):
        state = OrderedDict()
        state['name'] = self.name
        state['hist'] = normQuant(unp.nominal_values(self.hist),
                                  sigfigs=HASH_SIGFIGS)
        state['binning'] = self.binning._hashable_state
        stddevs = normQuant(unp.std_devs(self.hist), sigfigs=HASH_SIGFIGS)
        # TODO: better check here to see if the contained datatype is unp, as
        # opposed to 0 stddev (which could be the case but the user wants for
        # uncertainties to propagate)
        if np.all(stddevs == 0):
            stddevs = None
        else:
            stddevs = normQuant(stddevs, sigfigs=HASH_SIGFIGS)
        state['error_hist'] = stddevs
        state['tex'] = self.tex
        state['full_comparison'] = self.full_comparison
        return state

    def to_json(self, filename, **kwargs):
        """Serialize the state to a JSON file that can be instantiated as a new
        object later.

        Parameters
        ----------
        filename : str
            Filename; must be either a relative or absolute path (*not
            interpreted as a PISA resource specification*)
        **kwargs
            Further keyword args are sent to `pisa.utils.jsons.to_json()`

        See Also
        --------
        from_json : Intantiate new object from the file written by this method
        pisa.utils.jsons.to_json

        """
        jsons.to_json(self._serializable_state, filename=filename, **kwargs)

    @classmethod
    def from_json(cls, resource):
        """Instantiate a new Map object from a JSON file.

        The format of the JSON is generated by the `Map.to_json` method, which
        converts a Map object to basic types and then numpy arrays are
        converted in a call to `pisa.utils.jsons.to_json`.

        Parameters
        ----------
        resource : str
            A PISA resource specification (see pisa.utils.resources)

        See Also
        --------
        to_json
        pisa.utils.jsons.to_json

        """
        state = jsons.from_json(resource)
        # State is a dict with kwargs, so instantiate with double-asterisk
        # syntax
        return cls(**state)

    def assert_compat(self, other):
        if np.isscalar(other) or type(other) is uncertainties.core.Variable:
            return
        elif isinstance(other, np.ndarray):
            self.binning.assert_array_fits(other)
        elif isinstance(other, Map):
            self.binning.assert_compat(other.binning)
        else:
            raise TypeError('Unhandled type %s' %type(other))

    def iterbins(self):
        """Returns a bin iterator which yields a map containing a single bin
        each time. Modifications to that map will be reflected in this (the
        parent) map.

        Note that the returned map has the attribute `parent_indexer` for
        indexing directly into to the parent map (or to a similar map).

        Yields
        ------
        Map object containing one of each bin of this Map

        """
        shape = self.shape
        for i in xrange(self.hist.size):
            idx_item = np.unravel_index(i, shape)
            idx_view = [slice(x, x+1) for x in idx_item]
            single_bin_map = Map(
                name=self.name, hist=self.hist[idx_view],
                binning=self.binning[idx_item], hash=None, tex=self.tex,
                full_comparison=self.full_comparison
            )
            single_bin_map.parent_indexer = idx_item
            yield single_bin_map

    # TODO : example!
    def iterindices(self):
        """Iterator that yields the index for accessing each bin in
        the map.

        Examples
        --------
        >>> map = Map('x', binning=[dict('E', )])

        """
        shape = self.shape
        for i in xrange(self.hist.size):
            idx_item = np.unravel_index(i, shape)
            yield idx_item

    def __repr__(self):
        return np.array_repr(self._hist)

    def __hash__(self):
        if self.hash is not None:
            return self.hash
        raise ValueError('No hash defined.')

    def __setattr__(self, attr, value):
        """Only allow setting attributes defined in slots"""
        if attr not in self._slots:
            raise ValueError('Attribute "%s" not allowed to be set.' % attr)
        super(Map, self).__setattr__(attr, value)

    def __getattr__(self, attr):
        return super(Map, self).__getattribute__(attr)

    @new_obj
    def _slice_or_index(self, idx):
        """Slice or index into the map. Indexing single element in self.hist
        e.g. hist[1,3] returns a 0D array while hist[1,3:8] returns a 1D array,
        but we need 2D (in his example)... so reshape after indexing (the
        indexed binning obj implements this logic and so knows the shape the
        hist should be).

        """
        new_binning = self.binning[idx]
        return {'binning': self.binning[idx],
                'hist': np.reshape(self.hist[idx], new_binning.shape)}

    def __getitem__(self, idx):
        return self._slice_or_index(idx)

    def llh(self, expected_values):
        """Calculate the total log-likelihood value between this map and the map
        described by `expected_values`; self is taken to be the "actual values"
        (or (pseudo)data), and `expected_values` are the expectation values for
        each bin.

        Parameters
        ----------
        expected_values : numpy.ndarray or Map of same dimension as this

        Returns
        -------
        total_llh : float

        """
        if isinstance(expected_values, Map):
            expected_values = expected_values.hist
        return np.sum(llh(actual_values=self.hist,
                          expected_values=expected_values))

    def conv_llh(self, expected_values):
        """Calculate the total convoluted log-likelihood value between this map
        and the map described by `expected_values`; self is taken to be the
        "actual values" (or (pseudo)data), and `expected_values` are the
        expectation values for each bin.

        Parameters
        ----------
        expected_values : numpy.ndarray or Map of same dimension as this

        Returns
        -------
        total_conv_llh : float

        """
        if isinstance(expected_values, Map):
            expected_values = expected_values.hist
        return np.sum(conv_llh(actual_values=self.hist,
                               expected_values=expected_values))

    def mod_chi2(self, expected_values):
        """Calculate the total modified chi2 value between this map and the map
        described by `expected_values`; self is taken to be the "actual values"
        (or (pseudo)data), and `expected_values` are the expectation values for
        each bin.

        Parameters
        ----------
        expected_values : numpy.ndarray or Map of same dimension as this

        Returns
        -------
        total_mod_chi2 : float

        """
        if isinstance(expected_values, Map):
            expected_values = expected_values.hist
        return np.sum(mod_chi2(actual_values=self.hist,
                          expected_values=expected_values))

    def chi2(self, expected_values):
        """Calculate the total chi-squared value between this map and the map
        described by `expected_values`; self is taken to be the "actual values"
        (or (pseudo)data), and `expected_values` are the expectation values for
        each bin.

        Parameters
        ----------
        expected_values : numpy.ndarray or Map of same dimension as this

        Returns
        -------
        total_chi2 : float

        """
        if isinstance(expected_values, Map):
            expected_values = expected_values.hist
        return np.sum(chi2(actual_values=self.hist,
                           expected_values=expected_values))

    def __setitem__(self, idx, val):
        return setitem(self.hist, idx, val)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        assert isinstance(value, basestring)
        return super(Map, self).__setattr__('_name', value)

    @property
    def tex(self):
        return self._tex

    @tex.setter
    def tex(self, value):
        assert isinstance(value, basestring)
        return super(Map, self).__setattr__('_tex', value)

    @property
    def hash(self):
        return self._hash

    @hash.setter
    def hash(self, value):
        """Hash must be an immutable type (i.e., have a __hash__ method)"""
        assert hasattr(value, '__hash__')
        super(Map, self).__setattr__('_hash', value)

    @property
    def hist(self):
        return self._hist

    @property
    def nominal_values(self):
        return unp.nominal_values(self._hist)

    @property
    def std_devs(self):
        return unp.std_devs(self._hist)

    @property
    def binning(self):
        return self._binning

    @property
    def full_comparison(self):
        """Compare element-by-element instead of just comparing hashes."""
        return self._full_comparison

    @full_comparison.setter
    def full_comparison(self, value):
        assert isinstance(value, bool)
        super(Map, self).__setattr__('_full_comparison', value)

    # Common mathematical operators

    @new_obj
    def __abs__(self):
        state_updates = {
            #'name': "|%s|" % (self.name,),
            #'tex': r"{\left| %s \right|}" % strip_outer_parens(self.tex),
            'hist': np.abs(self.hist)
        }
        return state_updates

    @new_obj
    def __add__(self, other):
        """Add `other` to self"""
        if np.isscalar(other) or type(other) is uncertainties.core.Variable:
            state_updates = {
                #'name': "(%s + %s)" % (self.name, other),
                #'tex': r"{(%s + %s)}" % (self.tex, other),
                'hist': self.hist + other
            }
        elif isinstance(other, np.ndarray):
            state_updates = {
                #'name': "(%s + array)" % self.name,
                #'tex': r"{(%s + X)}" % self.tex,
                'hist': self.hist + other
            }
        elif isinstance(other, Map):
            state_updates = {
                #'name': "(%s + %s)" % (self.name, other.name),
                #'tex': r"{(%s + %s)}" % (self.tex, other.tex),
                'hist': self.hist + other.hist,
                'full_comparison': (self.full_comparison or
                                    other.full_comparison),
            }
        else:
            type_error(other)
        return state_updates

    #def __cmp__(self, other):

    @new_obj
    def __div__(self, other):
        if np.isscalar(other) or type(other) is uncertainties.core.Variable:
            state_updates = {
                #'name': "(%s / %s)" % (self.name, other),
                #'tex': r"{(%s / %s)}" % (self.tex, other),
                'hist': self.hist / other
            }
        elif isinstance(other, np.ndarray):
            state_updates = {
                #'name': "(%s / array)" % self.name,
                #'tex': r"{(%s / X)}" % self.tex,
                'hist': self.hist / other
            }
        elif isinstance(other, Map):
            state_updates = {
                #'name': "(%s / %s)" % (self.name, other.name),
                #'tex': r"{(%s / %s)}" % (self.tex, other.tex),
                'hist': self.hist / other.hist,
                'full_comparison': (self.full_comparison or
                                    other.full_comparison),
            }
        else:
            type_error(other)
        return state_updates

    def __truediv__(self, other):
        return self.__div__(other)

    def __floordiv__(self, other):
        raise NotImplementedError('floordiv not implemented for type Map')

    # TODO: figure out what we actually want to overload "==" with, and how
    # to implement all the possible kinds of "==" that might be useful for the
    # user, possibly with different methods altogether
    def __eq__(self, other):
        """Check if full state of maps are equal. *Not* element-by-element
        equality as for a numpy array. Call this.hist == other.hist for the
        nominal value and the error

        If `full_comparison` is true for *both* maps, or if either map lacks a
        hash, performs a full comparison of the contents of each map.

        Otherwise, simply checks that the hashes are equal.
        """
        if np.isscalar(other):
            # in case comparing with just with a scalar ignore the errors:
            return np.all(unp.nominal_values(self.hist) == other)
        if type(other) is uncertainties.core.Variable \
                or isinstance(other, np.ndarray):
            return (np.all(unp.nominal_values(self.hist)
                           == unp.nominal_values(other))
                    and np.all(unp.std_devs(self.hist)
                               == unp.std_devs(other)))
        elif isinstance(other, Map):
            if (self.full_comparison or other.full_comparison
                or self.hash is None or other.hash is None):
                return recursiveEquality(self._hashable_state,
                                         other._hashable_state)
            return self.hash == other.hash
        else:
            type_error(other)

    @new_obj
    def log(self):
        state_updates = {
            #'name': "log(%s)" % self.name,
            #'tex': r"\ln\left( %s \right)" % self.tex,
            'hist': np.log(self.hist)
        }
        return state_updates

    @new_obj
    def log10(self):
        state_updates = {
            #'name': "log10(%s)" % self.name,
            #'tex': r"\log_{10}\left( %s \right)" % self.tex,
            'hist': np.log10(self.hist)
        }
        return state_updates

    @new_obj
    def __mul__(self, other):
        if np.isscalar(other) or type(other) is uncertainties.core.Variable:
            state_updates = {
                #'name': "%s * %s" % (other, self.name),
                #'tex': r"%s \cdot %s" % (other, self.tex),
                'hist': self.hist * other
            }
        elif isinstance(other, np.ndarray):
            state_updates = {
                #'name': "array * %s" % self.name,
                #'tex': r"X \cdot %s" % self.tex,
                'hist': self.hist * other,
            }
        elif isinstance(other, Map):
            state_updates = {
                #'name': "%s * %s" % (self.name, other.name),
                #'tex': r"%s \cdot %s" % (self.tex, other.tex),
                'hist': self.hist * other.hist,
                'full_comparison': (self.full_comparison or
                                    other.full_comparison),
            }
        else:
            type_error(other)
        return state_updates

    def __ne__(self, other):
        return not self == other

    @new_obj
    def __neg__(self):
        state_updates = {
            #'name': "-%s" % self.name,
            #'tex': r"-%s" % self.tex,
            'hist': -self.hist,
        }
        return state_updates

    @new_obj
    def __pow__(self, other):
        if np.isscalar(other) or type(other) is uncertainties.core.Variable:
            state_updates = {
                #'name': "%s**%s" % (self.name, other),
                #'tex': "%s^{%s}" % (self.tex, other),
                'hist': np.power(self.hist, other)
            }
        elif isinstance(other, np.ndarray):
            state_updates = {
                #'name': "%s**(array)" % self.name,
                #'tex': r"%s^{X}" % self.tex,
                'hist': np.power(self.hist, other),
            }
        elif isinstance(other, Map):
            state_updates = {
                #'name': "%s**(%s)" % (self.name, strip_outer_parens(other.name)),
                #'tex': r"%s^{%s}" % (self.tex, strip_outer_parens(other.tex)),
                'hist': np.power(self.hist, other.hist),
                'full_comparison': (self.full_comparison or
                                    other.full_comparison),
            }
        else:
            type_error(other)
        return state_updates

    def __radd__(self, other):
        return self + other

    def __rdiv__(self, other):
        if isinstance(other, Map):
            return other / self
        else:
            return self.__rdiv(other)

    @new_obj
    def __rdiv(self,oher):
        if np.isscalar(other) or type(other) is uncertainties.core.Variable:
            state_updates = {
                #'name': "(%s / %s)" % (other, self.name),
                #'tex': "{(%s / %s)}" % (other, self.tex),
                'hist': other / self.hist,
            }
        elif isinstance(other, np.ndarray):
            state_updates = {
                #'name': "array / %s" % self.name,
                #'tex': "{(X / %s)}" % self.tex,
                'hist': other / self.hist,
            }
        else:
            type_error(other)
        return state_updates

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        if isinstance(other, Map):
            return other - self
        else:
            return self.__rsub(other)

    @new_obj
    def __rsub(self, other):
        if np.isscalar(other) or type(other) is uncertainties.core.Variable:
            state_updates = {
                #'name': "(%s - %s)" % (other, self.name),
                #'tex': "{(%s - %s)}" % (other, self.tex),
                'hist': other - self.hist,
            }
        elif isinstance(other, np.ndarray):
            state_updates = {
                #'name': "(array - %s)" % self.name,
                #'tex': "{(X - %s)}" % self.tex,
                'hist': other - self.hist,
            }
        else:
            type_error(other)
        return state_updates

    @new_obj
    def sqrt(self):
        state_updates = {
            #'name': "sqrt(%s)" % self.name,
            #'tex': r"\sqrt{%s}" % self.tex,
            'hist': unp.sqrt(self.hist),
        }
        return state_updates

    @new_obj
    def __sub__(self, other):
        if np.isscalar(other) or type(other) is uncertainties.core.Variable:
            state_updates = {
                #'name': "(%s - %s)" % (self.name, other),
                #'tex': "{(%s - %s)}" % (self.tex, other),
                'hist': self.hist - other,
            }
        elif isinstance(other, np.ndarray):
            state_updates = {
                #'name': "(%s - array)" % self.name,
                #'tex': "{(%s - X)}" % self.tex,
                'hist': self.hist - other,
            }
        elif isinstance(other, Map):
            state_updates = {
                #'name': "%s - %s" % (self.name, other.name),
                #'tex': "{(%s - %s)}" % (self.tex, other.tex),
                'hist': self.hist - other.hist,
                'full_comparison': (self.full_comparison or
                                    other.full_comparison),
            }
        else:
            type_error(other)
        return state_updates

# TODO: instantiate individual maps from dicts if passed as such, so user
# doesn't have to instantiate each map. Also, check for name collisions with
# one another and with attrs (so that __getattr__ can retrieve the map by name)

# TODO: add docstrings

class MapSet(object):
    """
    Ordered set of event rate maps (aka histograms) defined over an arbitrary
    regluar hyper-rectangular binning.


    Parameters
    ----------
    maps : Map or sequence of Map

    name : string

    tex : string

    hash : immutable

    collate_by_name : bool
        If True, when this MapSet is passed alongside another MapSet to a
        function that operates on the maps, contained maps in each will be
        accessed by name. Hence, only maps with the same names will be operated
        on simultaneously.

        If false, the contained maps in each MapSet will be accessed by their
        order in each MapSet. This behavior is useful if maps are renamed
        through some operation but their order is maintained, and then
        comparisons are sought with their progenitors with the original
        (different) name.

    """
    __slots = ('_name', '_hash', 'hash')
    __state_attrs = ('name', 'maps', 'tex', 'hash', 'collate_by_name')
    def __init__(self, maps, name=None, tex=None, hash=None,
                 collate_by_name=True):
        maps_ = []
        for m in maps:
            if isinstance(m, Map):
                maps_.append(m)
            else:
                maps_.append(Map(**m))
        tex = (r'{\rm %s}' %name) if tex is None else tex
        super(MapSet, self).__setattr__('maps', maps_)
        super(MapSet, self).__setattr__('name', name)
        super(MapSet, self).__setattr__('tex', tex)
        super(MapSet, self).__setattr__('collate_by_name', collate_by_name)
        super(MapSet, self).__setattr__('collate_by_num', not collate_by_name)
        self.hash = hash

    def __repr__(self):
        argstrs = [('%s=%s' %item) for item in self._serializable_state]
        return '%s(%s)' %(self.__class__.__name__, ', '.join(argstrs))

    @property
    def _serializable_state(self):
        state = OrderedDict()
        state['maps'] = [m._serializable_state for m in self.maps]
        state['name'] = self.name
        state['tex'] = self.tex
        state['collate_by_name'] = self.collate_by_name
        return state

    def to_json(self, filename, **kwargs):
        """Serialize the state to a JSON file that can be instantiated as a new
        object later.

        Parameters
        ----------
        filename : str
            Filename; must be either a relative or absolute path (*not
            interpreted as a PISA resource specification*)
        **kwargs
            Further keyword args are sent to `pisa.utils.jsons.to_json()`

        See Also
        --------
        from_json : Intantiate new object from the file written by this method
        pisa.utils.jsons.to_json

        """
        jsons.to_json(self._serializable_state, filename=filename, **kwargs)

    @classmethod
    def from_json(cls, resource):
        """Instantiate a new MapSet object from a JSON file.

        The format of the JSON is generated by the `MapSet.to_json` method,
        which converts a MapSet object to basic types and then numpy arrays are
        converted in a call to `pisa.utils.jsons.to_json`.

        Parameters
        ----------
        resource : str
            A PISA resource specification (see pisa.utils.resources)

        See Also
        --------
        to_json
        pisa.utils.jsons.to_json

        """
        state = jsons.from_json(resource)
        # State is a dict for Map, so instantiate with double-asterisk syntax
        return cls(**state)

    def index(self, m):
        """Find map according to `x` and return the corresponding index.
        Accepts either an integer index or a map name to make interface
        consistent.

        Parameters
        ----------
        x : int, string, or Map
            Map, map name, or integer index of map in this MapSet. If a Map is
            passed, only its name is matched to the maps in this set.

        Returns
        -------
        integer index to the map

        """
        if isinstance(x, int):
            l = len(self)
            assert x >= -l and x < l
        elif isinstance(x, Map):
            x = self.names.index(x.name)
        elif isinstance(x, basestring):
            x = self.names.index(x)
        else:
            raise TypeError('Unhandled type "%s" for `x`' %type(x))
        return x

    def pop(self, *args): #x=None):
        """Remove a map and return it. If a value is passed, the map
        corresponding to `index(value)` is removed and returned instead.

        Parameters
        ----------
        x (optional) : int, string, or Map
            Map, map name, or integer index of map in this MapSet. If a Map is
            passed, only its name is matched to the maps in this set.

        Returns
        -------
        Map removed from this MapSet

        See Also
        --------
        list.pop

        """
        if len(args) == 0:
            m = self.maps.pop()
        elif len(args) == 1:
            idx = self.index(args[0])
            m = self.maps.pop(idx)
        else:
            raise ValueError('`pop` takes 0 or 1 argument; %d passed instead.'
                             %len(args))
        return m

    def combine_re(self, regexes):
        """For each regex passed, add contained maps whose names match.

        If a single regex is passed, the corresponding maps are combined and
        returned as a Map object. If a *sequence* of regexes is passed, each
        grouping is combined into a map separately and the resulting maps are
        populated into a new MapSet to be returned.

        Parameters
        ----------
        regexes : compiled regex, str representing a regex, or sequence thereof
            See Python module `re` for formatting.

        Returns
        -------
        Map : if single regex is passed
        MapSet : if multiple regexes are passed

        Raises
        ------
        ValueError if any of the passed regexes fail to match any map names.

        Notes
        -----
        If special characters are used in the regex, like a backslash, be sure
        to use a Python raw string (which does not interpret such special
        characters), by prefixing the string with an "r". E.g., the regex to
        match a period requires passing
            `regex=r'\.'`

        Examples
        --------
        Get total of trck and cscd maps, which are named with suffixes "trck"
        and "cscd", respectively.

        >>> total_trck_map = outputs.combine_re('.*trck')
        >>> total_cscd_map = outputs.combine_re('.*cscd')

        Get a MapSet with both of the above maps in it (and a single command)

        >>> total_pid_maps = outputs.combine_re(['.*trck', '.*cscd'])

        Strict name-checking, combine  nue_cc + nuebar_cc, including both
        cascades and tracks.

        >>> nue_cc_nuebar_cc_map = outputs.combine_re(
        ...     '^nue(bar){0,1}_cc_(cscd|trck)$')

        Lenient nue_cc + nuebar_cc including both cascades and tracks.

        >>> nue_cc_nuebar_cc_map = outputs.combine_re('nue.*_cc_.*')

        Total of all maps

        >>> total = outputs.combine_re('.*')

        See Also
        --------
        combine_wildcard : similar method but using wildcards (like filename
            matching in the Unix shell)

        References
        ----------
        re : Python module used for parsing regular expressions
            https://docs.python.org/2/library/re.html

        """
        if isinstance(regexes, (basestring, re._pattern_type)):
            regexes = [regexes]
        resulting_maps = []
        for regex in regexes:
            if hasattr(regex, 'pattern'):
                pattern = regex.pattern
            else:
                pattern = regex
            maps_to_combine = []
            for m in self.maps:
                if re.match(regex, m.name) is not None:
                    logging.debug('Map "%s" will be added...' %m.name)
                    maps_to_combine.append(m)
            if len(maps_to_combine) == 0:
                raise ValueError('No map names match `regex` "%s"' % pattern)
            if len(maps_to_combine) > 1:
                m = reduce(add, maps_to_combine)
            else:
                m = copy(maps_to_combine[0])
            # Attach a "reasonable" name to the map; the caller can do better,
            # but this at least gives the user an idea of what the map
            # represents
            m.name = sanitize_name(pattern)
            resulting_maps.append(m)
        if len(resulting_maps) == 1:
            return resulting_maps[0]
        return MapSet(resulting_maps)

    def combine_wildcard(self, expressions):
        """For each expression passed, add contained maps whose names match.
        Expressions can contain wildcards like those used in the Unix shell.

        Valid wildcards (from fnmatch docs, link below):
            "*" : matches everything
            "?" : mateches any single character
            "[`seq`]" : matches any character in `seq`
            "[!`seq`]" : matches any character not in `seq`

        If a single expression is passed, the matching maps are combined and
        returned as a Map object. If a *sequence* of expressions is passed,
        each grouping is combined into a map separately and the resulting maps
        are populated into a new MapSet to be returned.

        Parameters
        ----------
        expressions : string or sequence thereof
            See Python module `fnmatch` for more info.

        Examples
        --------
        >>> total_trck_map = outputs.combine_wildcard('*trck')
        >>> total_cscd_map = outputs.combine_wildcard('*cscd')
        >>> total_pid_maps = outpubs.combine_wildcard(['*trck', '*cscd'])
        >>> nue_cc_nuebar_cc_map = outputs.combine_wildcard('nue*_cc_*')
        >>> total = outputs.combine_wildcard('*')

        See Also
        --------
        combine_re : similar method but using regular expressions

        References
        ----------
        fnmatch : Python module used for parsing the expression with wildcards
            https://docs.python.org/2/library/fnmatch.html

        """
        if isinstance(expressions, basestring):
            expressions = [expressions]
        resulting_maps = []
        for expr in expressions:
            maps_to_combine = []
            for m in self.maps:
                if fnmatch(m.name, expr):
                    logging.debug('Map "%s" will be added...' %m.name)
                    maps_to_combine.append(m)
            if len(maps_to_combine) == 0:
                raise ValueError('No map names match `expr` "%s"' % expr)
            if len(maps_to_combine) > 1:
                m = reduce(add, maps_to_combine)
            else:
                m = copy(maps_to_combine[0])
            # Reasonable name for giving user an idea of what the map
            # represents
            m.name = sanitize_name(expr)
            resulting_maps.append(m)
        if len(resulting_maps) == 1:
            return resulting_maps[0]
        return MapSet(resulting_maps)

    def __eq__(self, other):
        return recursiveEquality(self._hashable_state, other._hashable_state)

    @property
    def name(self):
        return super(MapSet, self).__getattribute__('_name')

    @name.setter
    def name(self, name):
        return super(MapSet, self).__setattr__('_name', name)

    @property
    def hash(self):
        """Hash value of the map set is based upon the contained maps.
            * If all maps have the same hash value, this value is returned as
              the map set's hash
            * If one or more maps have different hash values, a list of the
              contained maps' hash values is hashed
            * If any contained map has None for hash value, the hash value of
              the map set is also None (i.e., invalid)

        """
        hashes = self.hashes
        if all([(h is not None and h == hashes[0]) for h in hashes]):
            return hashes[0]
        if all([(h is not None) for h in hashes]):
            return hash_obj(hashes)
        return None

    @hash.setter
    def hash(self, val):
        """Setting a hash to `val` for the map set sets the hash values of all
        contained maps to `val`."""
        if val is not None:
            [setattr(m, 'hash', val) for m in self.maps]

    @property
    def names(self):
        return [mp.name for mp in self]

    @property
    def hashes(self):
        return [mp.hash for mp in self]

    def hash_maps(self, map_names=None):
        if map_names is None:
            map_names = [m.name for m in self]
        hashes = [m.hash for m in self if m.name in map_names]
        if all([(h != None) for h in hashes]):
            return hash_obj(hashes)
        return None

    def collate_with_names(self, vals):
        ret_dict = OrderedDict()
        [setitem(ret_dict, name, val) for name, val in zip(self.names, vals)]
        return ret_dict

    def find_map(self, value):
        idx = None
        if isinstance(value, Map):
            pass
        elif isinstance(value, basestring):
            try:
                idx = self.names.index(value)
            except ValueError:
                pass
        if idx is None:
            raise ValueError('Could not find map name "%s" in %s' %
                             (value, self))
        return self[idx]

    def apply_to_maps(self, attr, *args, **kwargs):
        if len(kwargs) != 0:
            raise NotImplementedError('Keyword arguments are not handled')
        if hasattr(attr, '__name__'):
            attrname = attr.__name__
        else:
            attrname = attr
        do_not_have_attr = np.array([(not hasattr(mp, attrname))
                                     for mp in self.maps])
        if np.any(do_not_have_attr):
            missing_in_names = ', '.join(
                np.array(self.names)[do_not_have_attr]
            )
            num_missing = np.sum(do_not_have_attr)
            num_total = len(do_not_have_attr)
            raise AttributeError(
                'Maps %s (%d of %d maps in set) do not have attribute "%s"'
                %(missing_in_names, num_missing, num_total, attrname)
            )

        # Retrieve the corresponding callables from contained maps
        val_per_map = [getattr(mp, attr) for mp in self]

        if not all([hasattr(meth, '__call__') for meth in val_per_map]):
            # If all results are maps, populate a new map set & return that
            if all([isinstance(r, Map) for r in val_per_map]):
                return MapSet(val_per_map)

            # Otherwise put in an ordered dict with <name>: <val> pairs ordered
            # according to the map ordering in this map set
            return self.collate_with_names(val_per_map)

        # Rename for clarity
        method_name = attr
        method_per_map = val_per_map

        # Create a set of args for *each* map in this map set: If an arg is a
        # MapSet, convert that arg into the map in that set corresponding to
        # the same map in this set.
        args_per_map = []
        for map_num, mp in enumerate(self):
            map_name = mp.name
            this_map_args = []
            for arg in args:
                if (np.isscalar(arg) or
                        type(arg) is uncertainties.core.Variable or
                        isinstance(arg, (basestring, np.ndarray))):
                    this_map_args.append(arg)
                elif isinstance(arg, MapSet):
                    if self.collate_by_name:
                        this_map_args.append(arg[map_name])
                    elif self.collate_by_num:
                        this_map_args.append(arg[map_num])
                else:
                    raise TypeError('Unhandled arg %s / type %s' %
                                    (arg, type(arg)))
            args_per_map.append(this_map_args)

        # Make the method calls and collect returned values
        returned_vals = [meth(*args)
                         for meth, args in zip(method_per_map, args_per_map)]

        # If all results are maps, put them into a new map set & return
        if all([isinstance(r, Map) for r in returned_vals]):
            return MapSet(returned_vals)

        # If None returned by all, return a single None
        if all([(r is None) for r in returned_vals]):
            return None

        # Otherwise put into an ordered dict with name: val pairs
        return self.collate_with_names(returned_vals)

    def __str__(self):
        if self.name is not None:
            my_name = "'" + self.name + "'"
        else:
            my_name = super(MapSet, self).__repr__()
        return "MapSet %s containing maps %s" % (my_name, self.names)

    def __repr__(self):
        return str(self)

    def __contains__(self, name):
        return name in [m.name for m in self]

    #def __setattr__(self, attr, val):
    #    print '__setattr__ being accessed, attr = %s, val = %s' %(attr, val)
    #    if attr in MapSet.__slots:
    #        print 'attr "%s" found in MapSet.slots.' %attr
    #        object.__setattr__(self, attr, val)
    #    else:
    #        returned_vals = [setattr(mp, attr, val) for mp in self]
    #        if all([(r is None) for r in returned_vals]):
    #            return
    #        return self.collate_with_names(returned_vals)

    def __getattr__(self, attr):
        if attr in [m.name for m in self]:
            return self[attr]
        return self.apply_to_maps(attr)

    def __iter__(self):
        return iter(self.maps)

    def __len__(self):
        return len(self.maps)

    def __getitem__(self, item):
        """Retrieve a map by name or retrieve maps' histogram values by index
        or slice.

        If `item` is a string, retrieve map by name.
        If `item is an integer or one-dim slice, retrieve maps by sequence
        If `item` is length-2 tuple or two-dim slice, retrieve value(s) of all
            contained maps, each indexed by map[`item`]. The output is returned
            in an ordered dict with format {<map name>: <values>, ...}

        """
        if isinstance(item, basestring):
            return self.find_map(item)
        elif isinstance(item, (int, slice)):
            rslt = self.maps[item]
            if hasattr(rslt, '__len__') and len(rslt) > 1:
                return MapSet(rslt)
            return rslt
        elif isinstance(item, Sequence):
            if len(item) == 1:
                return self.maps[item]
            elif len(item) == 2:
                return MapSet([getitem(m, item) for m in self])
            else:
                raise IndexError('too many indices for 2D hist')
        #elif isinstance(item, Sequence):
        #    assert len(item) == 2, 'Maps are 2D, and so must be indexed as such'
        #    return self.collate_with_names([getitem(m, item) for m in self])
        else:
            raise TypeError('getitem does not support `item` of type %s'
                            % type(item))

    def __abs__(self):
        return self.apply_to_maps('__abs__')

    def __add__(self, val):
        return self.apply_to_maps('__add__', val)

    def __truediv__(self, val):
        return self.apply_to_maps('__truediv__', val)

    def __div__(self, val):
        return self.apply_to_maps('__div__', val)

    def log(self):
        return self.apply_to_maps('log')

    def log10(self):
        return self.apply_to_maps('log10')

    def __mul__(self, val):
        return self.apply_to_maps('__mul__', val)

    def __neg__(self):
        return self.apply_to_maps('__neg__')

    def __pow__(self, val):
        return self.apply_to_maps('__pow__', val)

    def __radd__(self, val):
        return self.apply_to_maps('__radd__', val)

    def __rdiv__(self, val):
        return self.apply_to_maps('__rdiv__', val)

    def __rmul__(self, val):
        return self.apply_to_maps('__rmul__', val)

    def __rsub__(self, val):
        return self.apply_to_maps('__rsub__', val)

    def sqrt(self):
        return self.apply_to_maps('sqrt')

    def __sub__(self, val):
        return self.apply_to_maps('__sub__', val)

    def reorder_dimensions(self, order):
        return MapSet([m.reorder_dimensions(order=order) for m in self.maps])

    def rebin(self, *args, **kwargs):
        return MapSet([m.rebin(*args, **kwargs) for m in self.maps])

    def downsample(self, *args, **kwargs):
        return MapSet([m.downsample(*args, **kwargs) for m in self.maps])

    def metric_per_map(self, expected_values, metric):
        metric = metric.lower() if isinstance(metric, basestring) else metric
        if metric in VALID_METRICS:
            return self.apply_to_maps(metric, expected_values)
        else:
            raise ValueError('`metric` "%s" not recognized; use one of %s.'
                             %(metric, VALID_METRICS))

    def metric_total(self, expected_values, metric):
        return np.sum(self.metric_per_map(expected_values, metric).values())

    def chi2_per_map(self, expected_values):
        return self.apply_to_maps('chi2', expected_values)

    def chi2_total(self, expected_values):
        return np.sum(self.chi2_per_map(expected_values))

    def fluctuate(self, method, random_state=None, jumpahead=0):
        """Add fluctuations to the maps in the set and return as a new MapSet.

        Parameters
        ----------
        method : None or string
        random_stae : None, numpy.random.RandomState, or seed spec

        """
        random_state = get_random_state(random_state=random_state,
                                        jumpahead=jumpahead)
        new_maps = [m.fluctuate(method=method, random_state=random_state)
                    for m in self]
        return MapSet(maps=new_maps, name=self.name, tex=self.tex, hash=None,
                      collate_by_name=self.collate_by_name)

    def llh_per_map(self, expected_values):
        return self.apply_to_maps('llh', expected_values)

    def llh_total(self, expected_values):
        return np.sum(self.llh(expected_values))

    def set_poisson_errors(self):
        return self.apply_to_maps('set_poisson_errors')

## Now dynamically add all methods from Map to MapSet that don't already exist
## in MapSet (and make these methods distribute to contained maps)
##for method_name, method in sorted(Map.__dict__.items()):
#add_methods = '''__abs__ __add__ __div__ __mul__ __neg__ __pow__ __radd__
#__rdiv__ __rmul__ __rsub__ __sub__'''.split()
#
#for method_name in add_methods:
#    #if not hasattr(method, '__call__') or method_name in MapSet.__dict__:
#    #    continue
#    disallowed = ('__getattr__', '__setattr__', '__getattribute__',
#                  '__getitem__', '__eq__', '__ne__', '__str__', '__repr__')
#    if method_name in disallowed:
#        continue
#    print 'adding method "%s" to MapSet as an apply func' % method_name
#    arg_str = ', *args' # if len(args) > 0 else ''
#    eval('def {method_name}(self{arg_str}):\n'
#         '    return self.apply_to_maps({method_name}{arg_str})'.format(method_name=method_name, arg_str=arg_str))
#    #f.__doc__ = 'Apply method %s to all contained maps' % method_name
#    #method = getattr(Map, method_name)
#    #if method.__doc__:
#    #    f.__doc__ += '... ' + method.__doc__
#    setattr(MapSet, method_name, MethodType(eval(method_name), None, MapSet))


# TODO: add tests for llh, chi2 methods
def test_Map():
    import os
    import shutil
    import tempfile
    from pisa.utils.jsons import to_json
    from pisa import ureg, Q_

    n_ebins = 10
    n_czbins = 5
    n_azbins = 2
    e_binning = OneDimBinning(name='energy', tex=r'E_\nu', num_bins=n_ebins,
                              domain=(1,80)*ureg.GeV, is_log=True)
    cz_binning = OneDimBinning(name='coszen', tex=r'\cos\,\theta',
                               num_bins=n_czbins, domain=(-1,0), is_lin=True)
    az_binning = OneDimBinning(name='azimuth', tex=r'\phi',
                               num_bins=n_azbins, domain=(0,2*np.pi), is_lin=True)
    # set directly unumpy array with errors
    #m1 = Map(name='x', hist=unp.uarray(np.ones((40,20)),np.sqrt(np.ones((40,20)))), binning=(e_binning, cz_binning))
    # or call init poisson error afterwards
    m1 = Map(name='x', hist=np.ones((n_ebins, n_czbins)), hash=23,
             binning=(e_binning, cz_binning))
    print "downsampling ====================="
    m2 = m1 * 2
    print m2.downsample(1)
    print m2.downsample(5)
    print m2.downsample(1, 1)
    print m2.downsample(1, 5)
    print m2.downsample(5, 5)
    print m2.downsample(10, 5)
    print m2.downsample(10, 5).binning
    print "===================== downsampling"

    assert m1.hash == 23
    m1.hash = 42
    assert m1.hash == 42
    m1.set_poisson_errors()
    # or no errors at all
    m2 = Map(name='y', hist=2*np.ones((n_ebins, n_czbins)),
             binning=(e_binning, cz_binning))
    m3 = Map(name='z', hist=4*np.ones((n_ebins, n_czbins, n_azbins)),
             binning=(e_binning, cz_binning, az_binning))

    print 'm3[0,0,0] =', m3[0,0,0]
    testdir = tempfile.mkdtemp()
    try:
        for m in [m1, m2, m1+m2, m1-m2, m1/m2, m1*m2]:
            m_file = os.path.join(testdir, m.name + '.json')
            m.to_json(m_file)
            m_ = Map.from_json(m_file)
            assert m_ == m, 'm=\n%s\nm_=\n%s' %(m, m_)
            to_json(m, m_file)
            m_ = Map.from_json(m_file)
            assert m_ == m, 'm=\n%s\nm_=\n%s' %(m, m_)
    finally:
        shutil.rmtree(testdir, ignore_errors=True)

    print m1, m1.binning
    print m2, m2.binning
    print m1.nominal_values
    print m1.std_devs
    r = m1 + m2
    # compare only nominal val
    assert r == 3
    print r
    print 'm1+m2=3:', r, r[0,0]
    r = m2 + m1
    # or compare including errors
    assert r == ufloat(3,1)
    print 'm2+m1=3:', r, r[0,0]
    r = 2*m1
    assert r == ufloat(2,2)
    print '2*m1=2:', r, r[0,0]
    r = (2*m1 + 8) / m2
    assert r == ufloat(5,1)
    print '(2*m1 + 8) / m2=5:', r, r.hist[0,0]
    #r[:,1] = 1
    #r[2,:] = 2
    print 'r[0:2,0:5].hist:', r[0:2,0:5].hist
    print 'r[0:2,0:5].binning:', r[0:2,0:5].binning
    r = m1 / m2
    assert r == ufloat(0.5,0.5)
    print r, '=', r[0,0]
    print [b.binning.energy.midpoints[0] for b in m1.iterbins()][0:2]

    # Test reorder_dimensions
    e_binning = OneDimBinning(
        name='energy', num_bins=2, is_log=True, domain=[1, 80]*ureg.GeV
    )
    cz_binning = OneDimBinning(
        name='coszen', num_bins=3, is_lin=True, domain=[-1, 1]
    )
    az_binning = OneDimBinning(
        name='azimuth', num_bins=4, is_lin=True,
        domain=[0, 2*np.pi]*ureg.rad
    )
    a = []
    for i in range(len(e_binning)):
        b = []
        for j in range(len(cz_binning)):
            c = []
            for k in range(len(az_binning)):
                c.append(i*100 + j*10 + k)
            b.append(c)
        a.append(b)
    a = np.array(a)
    m_orig = Map(name='orig', hist=deepcopy(a),
                 binning=[e_binning, cz_binning, az_binning])
    m_new = m_orig.reorder_dimensions(['azimuth', 'energy', 'coszen'])

    assert np.alltrue(m_orig[:,0,0].hist.flatten() ==
                      m_new[0,:,0].hist.flatten())
    assert np.alltrue(m_orig[0,:,0].hist.flatten() ==
                      m_new[0,0,:].hist.flatten())
    assert np.alltrue(m_orig[0,0,:].hist.flatten() ==
                      m_new[:,0,0].hist.flatten())

    for dim in m_orig.binning.names:
        assert m_orig[:,0,0].binning[dim] == m_new[0,:,0].binning[dim]
        assert m_orig[0,:,0].binning[dim] == m_new[0,0,:].binning[dim]
        assert m_orig[0,0,:].binning[dim] == m_new[:,0,0].binning[dim]

    deepcopy(m_orig)

    print '<< PASSED : test_Map >>'


# TODO: add tests for llh, chi2 methods
# TODO: make tests use assert rather than rely on printouts!
def test_MapSet():
    from itertools import permutations
    import os
    import shutil
    import tempfile
    from pisa.utils.jsons import to_json
    from pisa import ureg, Q_

    n_ebins = 6
    n_czbins = 3
    e_binning = OneDimBinning(name='energy', tex=r'E_\nu', num_bins=n_ebins,
                              domain=(1,80)*ureg.GeV, is_log=True)
    cz_binning = OneDimBinning(name='coszen', tex=r'\cos\,\theta',
                               num_bins=n_czbins, domain=(-1,0), is_lin=True)
    binning = MultiDimBinning([e_binning, cz_binning])
    m1 = Map(name='ones', hist=np.ones(binning.shape), binning=binning,
             hash='xyz')
    m1.set_poisson_errors()
    m2 = Map(name='twos', hist=2*np.ones(binning.shape), binning=binning,
             hash='xyz')
    ms01 = MapSet([m1, m2])
    print "downsampling ====================="
    print ms01.downsample(3)
    print "===================== downsampling"
    ms01 = MapSet((m1, m2), name='ms01')
    ms02 = MapSet((m1, m2), name='map set 1')
    ms1 = MapSet(maps=(m1, m2), name='map set 1', collate_by_name=True,
                 hash=None)
    assert np.all(ms1.combine_re(r'.*').hist == ms1.combine_wildcard('*').hist)
    assert np.all(ms1.combine_re(r'.*').hist == (ms1.ones + ms1.twos).hist)
    assert np.all(ms1.combine_re(r'^(one|two)s.*$').hist ==
                  ms1.combine_wildcard('*s').hist)
    assert np.all(ms1.combine_re(r'^(one|two)s.*$').hist == (ms1.ones +
                                                             ms1.twos).hist)
    print ms1.combine_re(r'^o').hist
    print ms1.combine_wildcard(r'o*').hist
    print ms1.combine_re(r'^o').hist - ms1.combine_wildcard(r'o*').hist
    print 'map sets equal after combining?', \
            ms1.combine_re(r'^o') == ms1.combine_wildcard(r'o*')
    print 'hist equal after combining?', \
            np.all(ms1.combine_re(r'^o').hist == \
                   ms1.combine_wildcard(r'o*').hist)
    assert np.all(ms1.combine_re(r'^o').hist == ms1.combine_wildcard('o*').hist)
    print '5', ms1.names
    assert np.all(ms1.combine_re(r'^o').hist == ms1.ones.hist)
    print '6', ms1.names
    try:
        ms1.combine_re('three')
    except ValueError:
        pass
    else:
        assert False
    try:
        ms1.combine_wildcard('three')
    except ValueError:
        pass
    else:
        assert False

    assert ms1.hash == 'xyz'
    assert ms1.name == 'map set 1'
    ms1.hash = 10
    assert ms1.hash == 10
    assert m1.hash == 10
    assert m2.hash == 10
    ms1.hash = -10
    assert ms1.hash == -10
    # "Remove" the hash from the MapSet...
    ms1.hash = None
    # ... but this should not remove hashes from the contained maps
    assert m1.hash == -10
    # ... and hashing on the MapSet should see that all contained maps have the
    # SAME hash val, and so should just return the hash value shared among them
    # all
    assert ms1.hash == -10
    # However changing a single map's hash means not all hashes are the same
    # for all maps...
    m1.hash = 40
    # ... so a hash should be computed from all contained hashes
    assert ms1.hash != 40 and ms1.hash != -10

    assert ms1.maps == [m1, m2]
    assert ms1.names == ['ones', 'twos']
    assert ms1.tex == r'{\rm map set 1}'
    # Check the Poisson errors
    assert np.all(unp.nominal_values(ms1[0].hist) == np.ones(binning.shape))
    assert np.all(unp.std_devs(ms1[0].hist) == np.ones(binning.shape))
    assert np.all(ms1[1].hist == 2*np.ones(binning.shape))
    print 'ms1[0:2].hist:', ms1[0:2].hist
    print 'ms1[0:2,0:2].hist:', ms1[0:2,0:2].hist
    assert np.all(ms1.apply_to_maps('__add__', 1).ones == 2)

    m1 = Map(name='threes', hist=3*np.ones((n_ebins,n_czbins)), binning=binning)
    m2 = Map(name='fours', hist=4*np.ones((n_ebins,n_czbins)), binning=binning)
    ms2 = MapSet(maps=(m1, m2), name='map set 2', collate_by_name=False)

    try:
        print ms1.__add__(ms2)
    except ValueError:
        pass
    else:
        raise Exception('Should have errored out!')

    m1 = Map(name='fives', hist=5*np.ones((n_ebins,n_czbins)), binning=binning)
    m2 = Map(name='sixes', hist=6*np.ones((n_ebins,n_czbins)), binning=binning)
    ms3 = MapSet(maps=(m1, m2), name='map set 3', collate_by_name=False)
    ms4 = MapSet(maps=(m1, m2), name='map set 3', collate_by_name=False)
    assert ms3 == ms4

    print 'ms2.maps:', ms2.maps
    print "(ms2 + ms3).names", (ms2 + ms3).names
    print "(ms2 + ms3)[0,0].hist", (ms2 + ms3)[0,0].hist
    print "ms1['ones'][0,0]:", ms1['ones'][0,0]
    print 'ms1.__mul__(2)[0,0]:', ms1.__mul__(2)[0,0]
    print '(ms1 * 2)[0,0]:', (ms1 * 2)[0,0]
    print 'ms1.__add__(ms1)[0,0]:', ms1.__add__(ms1)[0,0]
    print '(ms1 + ms1)[0,0]:', (ms1 + ms1)[0,0]
    print ms1.names
    print '(ms1/ ms1)[0,0]:', (ms1 / ms1)[0,0]
    print '(ms1/ms1 - 1)[0,0]:', (ms1/ms1 - 1)[0,0]
    #print "ms1.log10()['ones']:", ms1.log10()['ones']
    #print "ms1.log10()[0,0]['ones']:", ms1.log10()[0,0]['ones']
    #print 'np.log10(ms1):', np.log10(ms1)
    print '(ms1 * np.e).binning:', (ms1 * np.e).binning
    #print 'np.log(ms1 * np.e)[0][0,0]:', (np.log(ms1 * np.e))[0][0,0]
    #print 'np.sqrt(ms1)[0][0:4,0:2].hist:', np.sqrt(ms1)[0][0:4,0:2].hist
    print 'str(ms1)', str(ms1)
    print 'str(ms4)', str(ms4)
    print 'ms3', ms3
    print 'ms4', ms4

    testdir = tempfile.mkdtemp()
    try:
        for ms in [ms01, ms02, ms1, ms2, ms3, ms4]:
            ms_file = os.path.join(testdir, ms.name + '.json')
            ms.to_json(ms_file)
            ms_ = MapSet.from_json(ms_file)
            assert ms_ == ms, 'ms=\n%s\nms_=\n%s' %(ms, ms_)
            to_json(ms, ms_file)
            ms_ = MapSet.from_json(ms_file)
            assert ms_ == ms, 'ms=\n%s\nms_=\n%s' %(ms, ms_)
    finally:
        shutil.rmtree(testdir, ignore_errors=True)

    deepcopy(ms01)

    # Test reorder_dimensions (this just tests that it succeeds on the map set;
    # correctness of the reordering is tested in the unit test for Map)
    for ms in [ms01, ms02, ms1, ms2, ms3, ms4]:
        for p in permutations(ms[0].binning.dimensions):
            ms.reorder_dimensions(p)

    print '<< PASSED : test_MapSet >>'


if __name__ == "__main__":
    test_Map()
    test_MapSet()
