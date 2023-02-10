import pytest
from docs2stubs.type_normalizer import check_normalizer
from hamcrest import assert_that, equal_to
from .test_normalize import tcheck, ntcheck



def test_astropy():
    #ntcheck("(2,) tuple", "tuple", None)

    #ntcheck("(2,) tuple of slice object", "tuple", None)

    #ntcheck("numpy.flatiter", "flatiter", {'numpy': ['flatiter']})
    #ntcheck("2D `~numpy.ndarray`", "NDArray", {'numpy.typing': ['NDArray']})
    #ntcheck("`~astropy.wcs.WCS` or None", "WCS|None", {'astropy.wcs': ['WCS']})
    #ntcheck("array-like, `~astropy.units.Quantity`, or `~astropy.time.Time`", "ArrayLike|Quantity|Time", {'numpy.typing': ['ArrayLike'], 'astropy.units': ['Quantity'], 'astropy.time': ['Time']})
    ntcheck("Quantity-like ['redshift'], array-like, or `~numbers.Number`", "", None)
    ntcheck("`~astropy.coordinates.Angle`, optional, keyword-only", "Angle|None", None)
    ntcheck("`~astropy.units.Quantity` ['angular speed'], optional, keyword-only", "Quantity", None)
    ntcheck("`~astropy.coordinates.BaseRepresentation` subclass instance", "BaseRepresentation", None)
    ntcheck("`~astropy.coordinates.BaseDifferential` subclass, str, dict", "BaseDifferential|str|dict", None)


    ntcheck("unit-like", "unit", None)
    ntcheck("callable, ``'first_found'`` or ``None``", "Callable|Literal['first_found']|None", None)
    ntcheck("`~astropy.coordinates.BaseCoordinateFrame` or `~astropy.coordinates.SkyCoord`", "BaseCoordinateFrame|SkyCoord", None)
    ntcheck("`NDData`-like instance", "NDData", None)
    ntcheck("mapping or None (optional, keyword-only)", "Mapping|None", None)
    ntcheck("`NDData`-like instance", "NDData", None)
    ntcheck("float or scalar quantity-like ['frequency']", "float|Scalar", None)
    ntcheck("quantity-like ['energy', 'mass'] or array-like", "quanity|ArrayLike", None)
    ntcheck("xml iterable", "", None)
    ntcheck("{'standard', 'model', 'log', 'psd'}", "Literal['standard','model','log','psd']", None)
    ntcheck("str ('warn'|'error'|'silent')", "Literal[''warn','error','silent']", None)
    ntcheck("writable file-like", "FileLike", None)
    ntcheck("`~numpy.dtype`", "", None)
    ntcheck("float or np.float32/np.float64", "", None)
    ntcheck("numpy array", "ArrayLike", None)
    ntcheck("`astropy.units.format.Base` instance or str", "Base|str", None)
    ntcheck("`~astropy.units.Quantity` or subclass", "Quantity", None)

    ntcheck("bool (default=True)", "bool", None)
    ntcheck("str, `~astropy.cosmology.Cosmology` class, or None (optional, keyword-only)", "str|Cosmology|None", None)

    ntcheck("numpy ndarray, dict, list, table-like object", "NDArray|dict|list|table", None)

    ntcheck("column element, list, ndarray, slice or tuple", "", None)
    #ntcheck("number, `~astropy.units.Quantity`, `~astropy.units.function.logarithmic.LogQuantity`, or sequence of quantity-like.", \
    #        "num|Quantity|LogQuantity|Sequence[Quantity]", {'astropy.units.function.logarithmic': ['LogQuantity'], 'astropy.units': ['Quantity'], 'typing': ['Sequence']})
    #ntcheck("list of sequence, dict, or module", "list[Sequence]|dict|module", {'typing': ['Sequence']})
    #ntcheck("any type", "Any", {'Any': ['typing']})
    ntcheck("int > 0 or None", "int", None)

    assert(False)
    """
4#scalar, array-like, or `~astropy.units.Quantity`#Scalar|array-like|astropy.units.Quantity
4#`~astropy.time.Time` or iterable#astropy.time.Time|Iterable
3#dict of set of str#Unknown
3#{'std', 'mad_std'} or callable#Literal['std', 'mad_std']Unknown|
3#None or int or tuple of int#None|int|tuple[int, ...]
3#int or np.int32/np.int64#int|np.int32/np.int64
3#array-like, ndim=1#array-like|ndim=1
3#str (optional, keyword-only)#str|tuple[optional, keyword-only]
3#int, slice or sequence of int#int|slice|Sequence[int]
3#str, callable, list of (str or callable)#str|Callable|Unknown|tuple[str or callable]
3#numpy ndarray, list of list#Unknown|Sequence[list]
3#str or list of str or None#str|Sequence[str]|None
3#float or `~astropy.units.Quantity` ['length']#float|astropy.units.Quantity|tuple['length']
3#`~astropy.table.Table` or `~astropy.table.Row` or list thereof#astropy.table.Table|astropy.table.Row|Unknown
3#`~astropy.units.Quantity` or `~astropy.time.TimeDelta`#astropy.units.Quantity|astropy.time.TimeDelta
3#`~astropy.units.Quantity`, `~astropy.coordinates.EarthLocation`, str, or None; optional#astropy.units.Quantity|astropy.coordinates.EarthLocation|str|Unknown
3#int or array-like (int)#int|array-like|tuple[int]
3#bool or sequence of bool#bool|Sequence[bool]
3#bool or "update"#bool|Literal["update"]
3#float, array of float, or `~astropy.time.Time` object#float|Sequence[float]|Unknown
3#float or float array#float|Unknown
3#path-like, readable file-like, or callable#PathLike|Unknown|Callable
3#`~astropy.units.Quantity` or array-like#astropy.units.Quantity|array-like
3#unit-like or None#unit-like|None
3#`~astropy.coordinates.BaseCoordinateFrame` subclass#Unknown
3#str or list of tuple#str|Sequence[tuple]
2#dict or set of str#dict|set[str]
2#(N,) array of float#tuple[N,]Sequence[float]|
2#(N-1,) array of float#tuple[N-1,]Sequence[float]|
2#1D array#Unknown
2#1D array-like#Unknown
2#`numpy.ndarray` or `astropy.convolution.Kernel`#numpy.ndarray|astropy.convolution.Kernel
2#float or array-like['dimensionless'] or quantity-like#float|array-like|tuple['dimensionless']Unknown|
2#dict or None (optional, keyword-only)#dict|None|tuple[optional, keyword-only]
2#Quantity-like ['redshift'] or array-like#Quantity-like|tuple['redshift']Unknown|
2#tuple of ints#tuple[int, ...]
2#list, ndarray, or None#list|np.ndarray|None
2#`~numpy.dtype`-like#numpy.dtype-like
2#tuple or ()#Unknown|tuple[]
2#int or 0#int|Literal[0]
2#dict-like or None#dict-like|None
2#class or tuple thereof#class|Unknown
2#str or iterable of str#str|Unknown
2#str or list of str or list of column-like#str|Sequence[str]|Unknown
2#empty dict or None#Unknown|None
2#(2,) tuple of bool#tuple[2,]tuple[bool, ...]|
2#int, list, ndarray, or slice#int|list|np.ndarray|slice
2#tuple (x, y) of bools#tuple|tuple[x, y]Unknown|
2#`~astropy.table.Table` or subclass#astropy.table.Table|subclass
2#`str` or `set` of `str`#str|set[str]
2#sequence of `~astropy.units.UnitBase`#Sequence[astropy.units.UnitBase]
2#iterable of classes#Unknown
2#positional arguments#Unknown
2#number or `~astropy.units.Quantity`#number|astropy.units.Quantity
2#int, float, or scalar array-like#int|float|Unknown
2#`~astropy.units.Unit`, str, or tuple#astropy.units.Unit|str|tuple
2#set of `~astropy.units.Unit`#set[astropy.units.Unit]
2#numpy ndarray, list, number, str, or bytes#Unknown|list|number|str|bytes
2#numpy ndarray, list, or number; optional#Unknown|list|Unknown
2#{`~datetime.tzinfo`, None}#datetime.tzinfo|None
2#`~astropy.units.UnitBase` instance or str#astropy.units.UnitBase|str
2#ndarray or MaskedArray#np.ndarray|MaskedArray
2#float or `~astropy.units.Quantity` ['mass']#float|astropy.units.Quantity|tuple['mass']
2#float or `~astropy.units.Quantity` or None.#float|astropy.units.Quantity|None.
2#list or tuple of length 2#list|Unknown
2#list of numpy array#Unknown
2#list of numpy arrays#Unknown
2#int or tuple thereof#int|Unknown
2#number or tuple thereof#number|Unknown
2#`dict`-like object#Unknown
2#str or bytes#str|bytes
2#str or readable file-like#str|Unknown
2#sequence of `Column` or `ColDefs` or ndarray or `~numpy.recarray`#Sequence[Column]|ColDefs|np.ndarray|numpy.recarray
2#str or `HDUList`#str|HDUList
2#`~astropy.io.fits.Header` or string or bytes#astropy.io.fits.Header|str|bytes
2#array or ``astropy.io.fits.hdu.base.DELAYED``#array|astropy.io.fits.hdu.base.DELAYED
2#str, file-like or `pathlib.Path`#str|FileLike|pathlib.Path
2#str, file-like#str|FileLike
2#str, `HDUList` object, or ``HDU`` object#str|Unknown|Unknown
2#`astropy.table.Table`#astropy.table.Table
2#`~astropy.io.ascii.BaseInputter`#astropy.io.ascii.BaseInputter
2#`~astropy.io.ascii.BaseOutputter`#astropy.io.ascii.BaseOutputter
2#tuple, list of tuple#tuple|Sequence[tuple]
2#list [str]#list|tuple[str]
2#`~astropy.wcs.wcsapi.BaseHighLevelWCS`#astropy.wcs.wcsapi.BaseHighLevelWCS
2#`~astropy.wcs.wcsapi.BaseLowLevelWCS`#astropy.wcs.wcsapi.BaseLowLevelWCS
2#`slice` or `tuple` or `int`#slice|tuple|int
2#`.BaseLowLevelWCS`#BaseLowLevelWCS
2#`~astropy.time.Time` or None#astropy.time.Time|None
2#`~astropy.coordinates.BaseRepresentation` or None#astropy.coordinates.BaseRepresentation|None
2#float or `np.ndarray`#float|np.ndarray
2#`~astropy.coordinates.SphericalRepresentation`#astropy.coordinates.SphericalRepresentation
2#eraASTROM array#Unknown
2#(..., N, N) array-like#tuple[..., N, N]ArrayLike|
2#ndarray or `~astropy.units.Quantity` or `SpectralCoord`#np.ndarray|astropy.units.Quantity|SpectralCoord
2#float or `~astropy.units.Quantity` ['speed']#float|astropy.units.Quantity|tuple['speed']
2#scalar, array-like, or `~astropy.units.Quantity` ['angle']#Scalar|array-like|astropy.units.Quantity|tuple['angle']
2#str or other#str|other
2#`~astropy.units.Quantity` ['time'] or callable#astropy.units.Quantity|tuple['time']Unknown|
2#`~astropy.cosmology.Cosmology` or None#astropy.cosmology.Cosmology|None
2#`~astropy.coordinates.Angle`#astropy.coordinates.Angle
2#array-like or `~astropy.units.Quantity` ['frequency']#array-like|astropy.units.Quantity|tuple['frequency']
2#int (default=1)#int|tuple[default=1]
2#float or `~astropy.units.Quantity` ['frequency']#float|astropy.units.Quantity|tuple['frequency']
2#array-like, `~astropy.units.Quantity`, `~astropy.time.Time`, or `~astropy.time.TimeDelta`#array-like|astropy.units.Quantity|astropy.time.Time|astropy.time.TimeDelta
2#float or `~astropy.units.Quantity` ['dimensionless', 'time']#float|astropy.units.Quantity|tuple['dimensionless', 'time']
2#{'linear', 'sqrt', 'power', log', 'asinh'}#Literal['linear']|Literal['sqrt']|Literal['power']|log'|Literal['asinh']
2#:class:`~astropy.visualization.wcsaxes.WCSAxes`#astropy.visualization.wcsaxes.WCSAxes
2#tuple or `~astropy.units.Quantity` ['angle']#tuple|astropy.units.Quantity|tuple['angle']
2#{'both', 'major', 'minor'}#Literal['both', 'major', 'minor']
2#{'in', 'out'}#Literal['in', 'out']
2#float or str#float|str
2#str or tuple#str|tuple
2#:class:`astropy.visualization.BaseTransform`#astropy.visualization.BaseTransform
2#float or sequence(3)#float|Sequence|tuple[3]
2#:class:`astropy.visualization.BaseStretch`#astropy.visualization.BaseStretch
1#:class:`~astropy.samp.SAMPIntegratedClient` or :class:`~astropy.samp.SAMPClient`#astropy.samp.SAMPIntegratedClient|astropy.samp.SAMPClient
1#:class:`~astropy.samp.SAMPHubProxy`#astropy.samp.SAMPHubProxy
1#`numpy.ndarray` (bool)#numpy.ndarray|tuple[bool]
1#int or tuple of int#int|tuple[int, ...]
1#sequence of bool#Sequence[bool]
1#int or sequence of scalar#int|Sequence[Scalar]
1#{'root-n','root-n-0','pearson','sherpagehrels','frequentist-confidence', 'kraft-burrows-nousek'}#Literal['root-n','root-n-0','pearson','sherpagehrels','frequentist-confidence', 'kraft-burrows-nousek']
1#float or numpy.ndarray#float|numpy.ndarray
1#list-like#list-like
1#list of (3,) tuple#Unknown|tuple[3,]tuple|
1#str or object#str|Any
1#2D array#Unknown
1#2D or 1D array-like#2D|Unknown
1#float or 1D array-like#float|Unknown
1#array-like, one dimension#array-like|Unknown
1#float or array of float or `~astropy.units.Quantity`#float|Sequence[float]|astropy.units.Quantity
1#`~astropy.modeling.Fittable1DModel`#astropy.modeling.Fittable1DModel
1#`~astropy.modeling.Fittable2DModel`#astropy.modeling.Fittable2DModel
1#list or array#list|array
1#`~astropy.modeling.Model` or callable.#astropy.modeling.Model|callable.
1#{'integral', 'peak'}#Literal['integral', 'peak']
1#`astropy.convolution.Kernel`#astropy.convolution.Kernel
1#`astropy.convolution.Kernel`, float, or int#astropy.convolution.Kernel|float|int
1#{'add', 'sub', 'mul'}#Literal['add', 'sub', 'mul']
1#`~astropy.nddata.NDData` or array-like#astropy.nddata.NDData|array-like
1#`numpy.ndarray` or `~astropy.convolution.Kernel`#numpy.ndarray|astropy.convolution.Kernel
1#{'fill', 'wrap'}#Literal['fill', 'wrap']
1#callable or boolean#Callable|bool
1#complex type#Unknown
1#`convolve` or `convolve_fft`#convolve|convolve_fft
1#unit-like or None (optional, keyword-only)#unit-like|None|tuple[optional, keyword-only]
1#`~astropy.units.Equivalency` or sequence thereof#astropy.units.Equivalency|Unknown
1#callable[[object, object, Any], Any] or str (optional, keyword-only)#callable[|tuple[object, object, Any], Any]Unknown||tuple[optional, keyword-only]
1#callable[[type, type, Any], Any]#callable[|tuple[type, type, Any], Any]
1#`~astropy.cosmology.Cosmology` instance#astropy.cosmology.Cosmology
1#callable[[object, object, Any], Any] or None#callable[|tuple[object, object, Any], Any]|None
1#function or method#function|method
1#float or array-like['dimensionless']#float|array-like|tuple['dimensionless']
1#int or array-like#int|array-like
1#sequence or object array[sequence]#Sequence|Unknown|tuple[sequence]
1#bool or None or str, optional keyword-only#bool|None|str|Unknown
1#:class:`astropy.cosmology.Cosmology` class or None#Unknown|None
1#bool, optional keyword-only#bool|Unknown
1#mapping#mapping
1#`_CosmologyModel` subclass instance#Unknown
1#str, keyword-only#str|keyword-only
1#{function, scipy.LowLevelCallable}#function|scipy.LowLevelCallable
1#`~numbers.Number` or scalar ndarray#numbers.Number|Unknown
1#bool or None, optional keyword-only#bool|None|Unknown
1#`~astropy.cosmology.FLRW` subclass instance#Unknown
1#float or quantity-like ['redshift']#float|quantity-like|tuple['redshift']
1#None, str, or `~astropy.cosmology.Cosmology`#None|str|astropy.cosmology.Cosmology
1#str or None, optional keyword-only#str|None|Unknown
1#{'comoving', 'lookback', 'luminosity'} or None#Literal['comoving', 'lookback', 'luminosity']|None
1#{'comoving', 'lookback', 'luminosity'} or None (optional, keyword-only)#Literal['comoving', 'lookback', 'luminosity']Unknown|tuple[optional, keyword-only]|
1#None or `~astropy.units.Quantity` ['frequency']#None|astropy.units.Quantity|tuple['frequency']
1#`~astropy.table.Table` object#Unknown
1#Column or mixin column#Column|Unknown
1#N-d sequence#Unknown
1#None or dtype-like#None|dtype-like
1#object exposing buffer interface#Unknown
1#{'C', 'F'}#Literal['C', 'F']
1#{'C', 'F', 'A', 'K'}#Literal['C', 'F', 'A', 'K']
1#str or `astropy.units.UnitBase` instance#str|astropy.units.UnitBase
1#list, ndarray or None#list|np.ndarray|None
1#float, int, str, or None#float|int|str|None
1#scalar; optional#Unknown
1#bool or array-like#bool|array-like
1#Table object#Unknown
1#dict, str#dict|str
1#dict, list, tuple; optional#dict|list|Unknown
1#list, optional:#list|optional:
1#str or list#str|list
1#type or None#type|None
1#object (column-like sequence)#Any|tuple[column-like sequence]
1#np.dtype or None#np.dtype|None
1#list of object#Sequence[Any]
1#list of int or None#Sequence[int]|None
1#`~astropy.table.Column` or `~numpy.ndarray` or sequence#astropy.table.Column|numpy.ndarray|Sequence
1#slice or int or array of int#slice|int|Sequence[int]
1#int, dict#int|dict
1#Table or DataFrame or ndarray#Table|DataFrame|np.ndarray
1#table-like object or list or scalar#Unknown|list|Scalar
1#str, list of str, numpy array, or `~astropy.table.Table`#str|Sequence[str]|Unknown|astropy.table.Table
1#None, bool, str#None|bool|str
1#`~astropy.units.Quantity` ['angle', 'length']#astropy.units.Quantity|tuple['angle', 'length']
1#str or function#str|function
1#{'first', 'last', 'none'}#Literal['first', 'last', 'none']
1#list of Tables#Sequence[Tables]
1#List of tables#Sequence[tables]
1#str or list or tuple#str|list|tuple
1#str, list of str, `Table`, or Numpy array#str|Sequence[str]|Table|Unknown
1#Table or Numpy array of same length as col#Table|Unknown
1#list or int#list|int
1#str, dict#str|dict
1#object or str#Any|str
1#unicode, str or bytes#unicode|str|bytes
1#type, instance, or None#type|instance|None
1#bool (defaults to False)#bool|tuple[defaults to False]
1#list or tuple#list|tuple
1#col.info.dtype#col.info.dtype
1#Index#Index
1#tuple, slice#tuple|slice
1#list or ndarray#list|np.ndarray
1#tuple, list#tuple|list
1#int, str, tuple, or list#int|str|tuple|list
1#New values of the row elements.#Unknown
1#tuple of class#tuple[class, ...]
1#quantity-like or `~astropy.units.PhysicalType`-like#quantity-like|astropy.units.PhysicalType-like
1#number, `~numpy.ndarray`, `~astropy.units.Quantity` (sequence), or str#number|numpy.ndarray|astropy.units.Quantity|tuple[sequence]str|
1#~numpy.dtype#numpy.dtype
1#ndarray or tuple thereof#np.ndarray|Unknown
1#ndarray or scalar#np.ndarray|Scalar
1#Arguments#Arguments
1#file or str or Path#file|str|Path
1#str or Path#str|Path
1#str or `astropy.units.format.Base` instance or subclass#str|astropy.units.format.Base|subclass
1#list of `astropy.units.UnitBase` instances#Unknown
1#list of int#Sequence[int]
1#astropy.units.core.UnitBase#astropy.units.core.UnitBase
1#unit-like, tuple of unit-like, or `~astropy.units.StructuredUnit`#unit-like|Unknown|astropy.units.StructuredUnit
1#tuple of str, tuple or list; `~numpy.dtype`; or `~astropy.units.StructuredUnit`#tuple[str, ...]|tuple|Unknown|astropy.units.StructuredUnit
1#float, int, Rational, Fraction#float|int|Rational|Fraction
1#iterable of str#Unknown
1#number or array#number|array
1#`~numpy.ufunc`#numpy.ufunc
1#`~astropy.units.Quantity` or ndarray subclass#astropy.units.Quantity|Unknown
1#array or `~astropy.units.Quantity` or tuple#array|astropy.units.Quantity|tuple
1#`~astropy.units.Unit` or None, or tuple#astropy.units.Unit|None|tuple
1#`numpy.dtype`#numpy.dtype
1#`~astropy.units.Unit`, string, or tuple#astropy.units.Unit|str|tuple
1#`~astropy.units.Unit`, `~astropy.units.function.FunctionUnitBase`, or str#astropy.units.Unit|astropy.units.function.FunctionUnitBase|str
1#number, quantity-like, or sequence thereof#number|quantity-like|Unknown
1#list of equivalency pairs#Unknown
1#list of equivalency pairs, or None#Unknown|None
1#module#module
1#str, list of str, 2-tuple#str|Sequence[str]|2-tuple
1#sequence of `UnitBase`#Sequence[UnitBase]
1#sequence of numbers#Sequence[numbers]
1#int or float value, or sequence of such values#int|Unknown|Unknown
1#`~astropy.units.Quantity` ['solid angle']#astropy.units.Quantity|tuple['solid angle']
1#`~astropy.units.Quantity` ['temperature'] or None#astropy.units.Quantity|tuple['temperature']|None
1#sequence, ndarray, number, str, bytes, or `~astropy.time.Time` object#Sequence|np.ndarray|number|str|bytes|Unknown
1#sequence, ndarray, or number; optional#Sequence|np.ndarray|Unknown
1#`~astropy.coordinates.EarthLocation` or tuple#astropy.coordinates.EarthLocation|tuple
1#str, sequence, or ndarray#str|Sequence|np.ndarray
1#str or None; optional#str|Unknown
1#tuple of str#tuple[str, ...]
1#`~astropy.utils.iers.IERS`#astropy.utils.iers.IERS
1#sequence, ndarray, number, `~astropy.units.Quantity` or `~astropy.time.TimeDelta` object#Sequence|np.ndarray|number|astropy.units.Quantity|Unknown
1#sequence, ndarray, number, or `~astropy.units.Quantity`; optional#Sequence|np.ndarray|number|Unknown
1#dict or bool#dict|bool
1#ndarray or sequence#np.ndarray|Sequence
1#:class:`~astropy.modeling.Model` instance#astropy.modeling.Model
1#list or (2,) ndarray#Unknown|tuple[2,]np.ndarray|
1#`~astropy.units.Quantity` ['temperature']#astropy.units.Quantity|tuple['temperature']
1#float or `~astropy.units.Quantity` ['dimensionless']#float|astropy.units.Quantity|tuple['dimensionless']
1#float, `~numpy.ndarray`, or `~astropy.units.Quantity` ['frequency']#float|numpy.ndarray|astropy.units.Quantity|tuple['frequency']
1#float, `~numpy.ndarray`, or `~astropy.units.Quantity`#float|numpy.ndarray|astropy.units.Quantity
1#float, `~numpy.ndarray`, or `~astropy.units.Quantity` ['dimensionless']#float|numpy.ndarray|astropy.units.Quantity|tuple['dimensionless']
1#tuple or str#tuple|str
1#:class:`~astropy.cosmology.Cosmology`#astropy.cosmology.Cosmology
1#`Fitter`#Fitter
1#list of `~importlib.metadata.EntryPoint`#Sequence[importlib.metadata.EntryPoint]
1#float or `~astropy.units.Quantity`, optional.#float|astropy.units.Quantity|optional.
1#callable or False#Callable|False
1#dict, tuple#dict|tuple
1#_SelectorArguments#_SelectorArguments
1#optional, callable#optional|Callable
1#ndarray of dtype np.bool#Unknown
1#bool or dict#bool|dict
1#`Model`#Model
1#tuple of ndarray of float#Unknown
1#list of scalar or list of ndarray#Sequence[Scalar]|Sequence[np.ndarray]
1#callable, None#Callable|None
1#iterable, None#Iterable|None
1#iterable. None#Unknown
1#int, str, list, None#int|str|list|None
1#bool, None#bool|None
1#int, str, list, None (default = 0)#int|str|list|None|tuple[default = 0]
1#bool, None (default = None)#bool|None|tuple[default = None]
1#int, bool (default = False)#int|bool|tuple[default = False]
1#data-type (default = ``numpy.bool_``)#data-type|tuple[default = ``numpy.bool_``]
1#tuple of int or int#tuple[int, ...]|int
1#ndarray or array-like#np.ndarray|array-like
1#tuple or `~astropy.coordinates.SkyCoord`#tuple|astropy.coordinates.SkyCoord
1#int, array-like, or `~astropy.units.Quantity`#int|array-like|astropy.units.Quantity
1#{'trim', 'partial', 'strict'}#Literal['trim', 'partial', 'strict']
1#`matplotlib.axes.Axes` instance#matplotlib.axes.Axes
1#`numpy.ndarray`-like or `NDData`-like#numpy.ndarray-like|NDData-like
1#`NDData` instance#NDData
1#`~astropy.units.Quantity` or ndarray#astropy.units.Quantity|np.ndarray
1#`numpy.ndarray` or number#numpy.ndarray|number
1#`~astropy.nddata.CCDData`-like or array-like#astropy.nddata.CCDData-like|array-like
1#`~astropy.nddata.StdDevUncertainty`,             `~astropy.nddata.VarianceUncertainty`,             `~astropy.nddata.InverseVariance`, `numpy.ndarray` or             None#astropy.nddata.StdDevUncertainty|astropy.nddata.VarianceUncertainty|astropy.nddata.InverseVariance|numpy.ndarray|None
1#`numpy.ndarray` or None#numpy.ndarray|None
1#`numpy.ndarray` or `~astropy.nddata.FlagCollection` or None#numpy.ndarray|astropy.nddata.FlagCollection|None
1#`~astropy.wcs.WCS` or None#astropy.wcs.WCS|None
1#dict-like object or None#Unknown|None
1#`~astropy.units.Unit` or str#astropy.units.Unit|str
1#astropy.io.fits.header or other dict-like#astropy.io.fits.header|Unknown
1#int, str, tuple of (str, int)#int|str|Unknown|tuple[str, int]
1#same type (class) as self#Unknown|tuple[class]Unknown|
1#``Number`` or `~numpy.ndarray`#Number|numpy.ndarray
1#`~astropy.units.Quantity` or `~numpy.ndarray`#astropy.units.Quantity|numpy.ndarray
1#`NDData` instance or subclass#NDData|subclass
1#instance or class#instance|class
1#`~astropy.nddata.NDData`-like#astropy.nddata.NDData-like
1#ndarray or `NDData`#np.ndarray|NDData
1#`~astropy.nddata.NDUncertainty`#astropy.nddata.NDUncertainty
1#array-like or `~astropy.nddata.FlagCollection`#array-like|astropy.nddata.FlagCollection
1#None#None
1#`astropy.units.UnitBase` instance or str#astropy.units.UnitBase|str
1#str or sequence of str#str|Sequence[str]
1#str or sequence of str or None#str|Sequence[str]|None
1#str or number or sequence of str or number#str|number|Sequence[str]|number
1#str or None, optional, keyword-only#str|None|optional|keyword-only
1#bool or `~astropy.units.Unit`#bool|astropy.units.Unit
1#int or `~astropy.units.Quantity`#int|astropy.units.Quantity
1#boolean#bool
1#iterable of str or None#Unknown|None
1#set of str or list of str or None#set[str]|Sequence[str]|None
1#float, array, or `~astropy.time.Time`#float|array|astropy.time.Time
1#path-like or None#PathLike|None
1#bool, or 'only', or 'empty'#bool|Literal['only']|Literal['empty']
1#str, `~numpy.dtype`, type#str|numpy.dtype|type
1#slice, int, list, or ndarray#slice|int|list|np.ndarray
1#int, list, or ndarray#int|list|np.ndarray
1#`~astropy.table.Column` or mixin#astropy.table.Column|mixin
1#slice, list, or ndarray#slice|list|np.ndarray
1#module or `str`#module|str
1#bool or list of str#bool|Sequence[str]
1#`type`#type
1#`object`#Any
1#sequence of types#Sequence[types]
1#dict of str -> str#Unknown
1#keyword args#Unknown
1#array-like of bool#Unknown
1#str or None, ignored.#str|None|ignored.
1#str or list of str.#str|Sequence[str.]
1#int or sequence of ints#int|Sequence[int]
1#{'introselect'}#Literal['introselect']
1#str or dtype#str|dtype
1#dtype object#Unknown
1#{'C', 'F', 'A', or 'K'}#Literal['C']|Literal['F']|Literal['A']|Literal['K']
1#bool, optional.#bool|optional.
1#int or sequence of ints, optional.#int|Sequence[int]|optional.
1#list of ndarray#Sequence[np.ndarray]
1#`~astropy.utils.metadata.MergeStrategy`#astropy.utils.metadata.MergeStrategy
1#collections.namedtuple#collections.namedtuple
1#sequence of str or dict of str keys#Sequence[str]|Unknown
1#numpy dtype object#Unknown
1#bytes#bytes
1#numpy bool array#Unknown
1#`~astropy.io.votable.tree.Field`#astropy.io.votable.tree.Field
1#astropy.io.votable.tree.Field#astropy.io.votable.tree.Field
1#Numpy dtype instance#Unknown
1#`astropy.table.Column` instance#astropy.table.Column
1#str, astropy.units.format.Base instance or None#str|astropy.units.format.Base|None
1#`~astropy.io.votable.tree.VOTableFile` or `~astropy.table.Table` instance.#astropy.io.votable.tree.VOTableFile|Unknown
1#str or writable file-like#str|Unknown
1#`~astropy.table.Table` instance#astropy.table.Table
1#str or sequence#str|Sequence
1#str or `~astropy.io.votable.tree.VOTableFile` or `~astropy.io.votable.tree.Table`#str|astropy.io.votable.tree.VOTableFile|astropy.io.votable.tree.Table
1#generator#generator
1#str or path-like or None#str|PathLike|None
1#file-like or None.#FileLike|None.
1#``_UnifiedIORegistryBase`` or None#_UnifiedIORegistryBase|None
1#None or path-like#None|PathLike
1#`~astropy.io.registry.UnifiedReadWrite` subclass#Unknown
1#sequence of `Column` or a `ColDefs`#Sequence[Column]|Unknown
1#`~astropy.table.Column`#astropy.table.Column
1#`~astropy.io.fits.header.Header`#astropy.io.fits.header.Header
1#file-like, string, or None#FileLike|str|None
1#list of int or str#Sequence[int]|str
1#or str#Unknown
1#str, bytearray, memoryview, ndarray#str|bytearray|memoryview|np.ndarray
1#int, callable#int|Callable
1#str, int, float, complex, bool, None#str|int|float|complex|bool|None
1#BaseHDU or sequence thereof#BaseHDU|Unknown
1#file-like, bytes#FileLike|bytes
1#str, buffer-like, etc.#str|buffer-like|etc.
1#int, str, tuple of (string, int)#int|str|Unknown|tuple[string, int]
1#int, str, tuple of (string, int) or BaseHDU#int|str|Unknown|tuple[string, int]Unknown|
1#file-like or bool#FileLike|bool
1#sequence of `Column`, `ColDefs` -like#Sequence[Column]|Unknown
1#array or `FITS_rec`#array|FITS_rec
1#array, `FITS_rec`, or `~astropy.table.Table`#array|FITS_rec|astropy.table.Table
1#HDUList#HDUList
1#str or file-like or compatible `astropy.io.fits` HDU object#str|FileLike|Unknown
1#str, int, float#str|int|float
1#array or `~numpy.recarray` or `~astropy.io.fits.Group`#array|numpy.recarray|astropy.io.fits.Group
1#astropy.table.Table#astropy.table.Table
1#array, :class:`~astropy.table.Table`, or `~astropy.io.fits.Group`#array|astropy.table.Table|astropy.io.fits.Group
1#array, `~astropy.table.Table`, or `~astropy.io.fits.Group`#array|astropy.table.Table|astropy.io.fits.Group
1#file, bool#file|bool
1#dictionary#dict
1#hashable#hashable
1#A ``Table.Column`` object.#Unknown
1#numpy data-type#Unknown
1#`~astropy.table.Table`, `~astropy.io.ascii.BaseHeader`#astropy.table.Table|astropy.io.ascii.BaseHeader
1#`~astropy.io.ascii.BaseReader`#astropy.io.ascii.BaseReader
1#str, file-like, list, `pathlib.Path` object#str|FileLike|list|Unknown
1#str, `~astropy.io.ascii.BaseReader`#str|astropy.io.ascii.BaseReader
1#bool, str or dict#bool|str|dict
1#``Writer``#Writer
1#`~astropy.io.ascii.BaseReader`, array-like, str, file-like, list#astropy.io.ascii.BaseReader|array-like|str|FileLike|list
1#str, bool#str|bool
1#:class:`~astropy.table.Table`#astropy.table.Table
1#str or :class:`h5py.File` or :class:`h5py.Group` or#str|h5py.File|Unknown
1#str or :class:`h5py.File` or :class:`h5py.Group`#str|h5py.File|h5py.Group
1#bool or str or int#bool|str|int
1#`~pyarrow.NativeFile` or None#pyarrow.NativeFile|None
1#str or path-like or file-like object#str|PathLike|Unknown
1#list [tuple] or list [list [tuple] ] or None#list|tuple[tuple]Unknown||tuple[tuple] ]|None
1#str or path-like#str|PathLike
1#`dict`#dict
1#str or :class:`py.lath:local`#str|py.lath:local
1#str or :class:`py.path:local`#str|py.path:local
1#:class:`~astropy.wcs.WCS` instance#astropy.wcs.WCS
1#:class:`~astropy.coordinates.baseframe.BaseCoordinateFrame` subclass instance#Unknown
1#(`numpy.ndarray`, `numpy.ndarray`) tuple#tuple[`numpy.ndarray`, `numpy.ndarray`]tuple|
1#'center' or ~astropy.coordinates.SkyCoord`#Literal['center']|astropy.coordinates.SkyCoord
1#str or `~astropy.wcs.WCS`#str|astropy.wcs.WCS
1#None or int#None|int
1#`~astropy.io.fits.Header`, `~astropy.io.fits.hdu.image.PrimaryHDU`, `~astropy.io.fits.hdu.image.ImageHDU`, str, dict-like, or None#astropy.io.fits.Header|astropy.io.fits.hdu.image.PrimaryHDU|astropy.io.fits.hdu.image.ImageHDU|str|dict-like|None
1#`~astropy.io.fits.HDUList`#astropy.io.fits.HDUList
1#int or a sequence.#int|Unknown
1#int array#Unknown
1#`~astropy.io.fits.Header` object#Unknown
1#(int, int)#tuple[int, int]
1#str or `~astropy.io.fits.Header` object.#str|Unknown
1#str or file-like or `~astropy.io.fits.HDUList`#str|FileLike|astropy.io.fits.HDUList
1#`astropy.wcs.wcsapi.BaseLowLevelWCS`#astropy.wcs.wcsapi.BaseLowLevelWCS
1#str, or list of str#str|Sequence[str]
1#str or file-like or None#str|FileLike|None
1#`CartesianRepresentation`#CartesianRepresentation
1#dict, `~astropy.coordinates.BaseDifferential`#dict|astropy.coordinates.BaseDifferential
1#`~astropy.coordinates.Longitude` or float#astropy.coordinates.Longitude|float
1#`~astropy.coordinates.Latitude` or float#astropy.coordinates.Latitude|float
1#`~astropy.units.Quantity` ['length'] or float#astropy.units.Quantity|tuple['length']Unknown|
1#dict[str, `~astropy.units.Quantity`]#dict|tuple[str, `~astropy.units.Quantity`]
1#ndarray or `~astropy.units.Quantity` or `SpectralQuantity`#np.ndarray|astropy.units.Quantity|SpectralQuantity
1#list of `~astropy.units.equivalencies.Equivalency`#Sequence[astropy.units.equivalencies.Equivalency]
1#{'relativistic', 'optical', 'radio'}#Literal['relativistic', 'optical', 'radio']
1#`~.coordinates.TransformGraph`#coordinates.TransformGraph
1#An `~astropy.utils.iers.IERSRangeError`#Unknown
1#Time object#Unknown
1#str or array-like#str|array-like
1#str, `~astropy.coordinates.BaseCoordinateFrame` or `~astropy.coordinates.SkyCoord`#str|astropy.coordinates.BaseCoordinateFrame|astropy.coordinates.SkyCoord
1#`~astropy.units.Quantity` or `~astropy.coordinates.CartesianDifferential`#astropy.units.Quantity|astropy.coordinates.CartesianDifferential
1#sequence of `~astropy.coordinates.BaseRepresentation`#Sequence[astropy.coordinates.BaseRepresentation]
1#sequence of coordinate-like#Unknown
1#`~astropy.coordinates.CartesianRepresentation`#astropy.coordinates.CartesianRepresentation
1#bool or str or KDTree#bool|str|KDTree
1#unit or None#unit|None
1#`~numpy.array`, scalar, `~astropy.units.Quantity`, :class:`~astropy.coordinates.Angle`#numpy.array|Scalar|astropy.units.Quantity|astropy.coordinates.Angle
1#`~astropy.units.UnitBase`#astropy.units.UnitBase
1#array, list, scalar, `~astropy.units.Quantity`, `~astropy.coordinates.Angle`#array|list|Scalar|astropy.units.Quantity|astropy.coordinates.Angle
1#tuple or angle-like#tuple|angle-like
1#unit-like ['angle']#unit-like|tuple['angle']
1#`astropy.time.Time` or str#astropy.time.Time|str
1#number, quantity-like#number|quantity-like
1#`CoordinateTransform`#CoordinateTransform
1#None or str#None|str
1#`TransformGraph` object#Unknown
1#a TransformGraph object#Unknown
1#array-like or callable#array-like|Callable
1#sequence of `CoordinateTransform` object#Unknown
1#str or `~astropy.coordinates.BaseRepresentation` subclass#str|Unknown
1#dict of str or `~astropy.coordinates.BaseDifferentials`#Unknown|astropy.coordinates.BaseDifferentials
1#('base', 's', `None`)#tuple['base', 's', `None`]
1#str, `~astropy.coordinates.BaseRepresentation` subclass#str|Unknown
1#`~astropy.coordinates.BaseDifferential` subclass#Unknown
1#`~astropy.coordinates.BaseRepresentation`#astropy.coordinates.BaseRepresentation
1#subclass of BaseRepresentation or string#Unknown|str
1#subclass of `~astropy.coordinates.BaseDifferential`, str#Unknown|str
1#bool, keyword-only#bool|keyword-only
1#coordinate-like or `BaseCoordinateFrame` subclass instance#coordinate-like|Unknown
1#`BaseCoordinateFrame` subclass or instance#Unknown|instance
1#:class:`~astropy.coordinates.BaseCoordinateFrame`#astropy.coordinates.BaseCoordinateFrame
1#number or `~astropy.units.Quantity` or None#number|astropy.units.Quantity|None
1#`~astropy.coordinates.BaseCoordinateFrame` class#Unknown
1#scalar or `~astropy.units.Quantity` ['length']#Scalar|astropy.units.Quantity|tuple['length']
1#`~astropy.units.UnitBase` ['length']#astropy.units.UnitBase|tuple['length']
1#`~astropy.units.Quantity` or `~astropy.coordinates.Angle`#astropy.units.Quantity|astropy.coordinates.Angle
1#`~astropy.coordinates.BaseCoordinateFrame` class or string#Unknown|str
1#`~astropy.units.Unit`, string, or tuple of :class:`~astropy.units.Unit` or str#astropy.units.Unit|str|Unknown|str
1#str or Representation class#str|Unknown
1#frame class, frame object, or str#Unknown|Unknown|str
1#str, `BaseCoordinateFrame` class or instance, or `SkyCoord` instance#str|Unknown|instance|SkyCoord
1#`~astropy.units.Quantity`, `~astropy.time.TimeDelta`#astropy.units.Quantity|astropy.time.TimeDelta
1#{'hmsdms', 'dms', 'decimal'}#Literal['hmsdms', 'dms', 'decimal']
1#SkyCoord or BaseCoordinateFrame#SkyCoord|BaseCoordinateFrame
1#`SkyCoord`#SkyCoord
1#`~astropy.coordinates.EarthLocation` or None#astropy.coordinates.EarthLocation|None
1#str or `BaseCoordinateFrame` class or instance#str|Unknown|instance
1#`~astropy.time.TimeDelta` or `~astropy.units.Quantity`#astropy.time.TimeDelta|astropy.units.Quantity
1#bool (default = False)#bool|tuple[default = False]
1#array-like, shape=(n_times,)#array-like|shape=|tuple[n_times,]
1#int (default = 5)#int|tuple[default = 5]
1#float, array, or None#float|array|None
1#array-like, `~astropy.units.Quantity`, or `~astropy.time.Time` (optional)#array-like|astropy.units.Quantity|astropy.time.Time|tuple[optional]
1#str or float or `~astropy.units.Quantity`#str|float|astropy.units.Quantity
1#{'likelihood', 'snr'}#Literal['likelihood', 'snr']
1#{'fast', 'slow'}#Literal['fast', 'slow']
1#array-like, `~astropy.units.Quantity`, or `~astropy.time.Time`#array-like|astropy.units.Quantity|astropy.time.Time
1#numpy ndarray, dict, list, `~astropy.table.Table`, or table-like object#Unknown|dict|list|astropy.table.Table|Unknown
1#`~astropy.time.Time`, `~astropy.time.TimeDelta` or iterable#astropy.time.Time|astropy.time.TimeDelta|Iterable
1#`~astropy.time.Time` or str#astropy.time.Time|str
1#`~astropy.time.TimeDelta` or `~astropy.units.Quantity` ['time']#astropy.time.TimeDelta|astropy.units.Quantity|tuple['time']
1#`~astropy.units.Quantity` ['time']#astropy.units.Quantity|tuple['time']
1#`str` or `pathlib.Path`#str|pathlib.Path
1#:class:`~astropy.timeseries.TimeSeries`#astropy.timeseries.TimeSeries
1#`~astropy.units.Quantity` or `~astropy.time.TimeDelta` ['time']#astropy.units.Quantity|astropy.time.TimeDelta|tuple['time']
1#`~astropy.visualization.BaseInterval` subclass instance#Unknown
1#`~astropy.visualization.BaseStretch` subclass instance#Unknown
1#2D or 3D array-like#2D|Unknown
1#None or `~matplotlib.axes.Axes`#None|matplotlib.axes.Axes
1#`~matplotlib.path.Path`#matplotlib.path.Path
1#{ 'lines' | 'contours' }#Unknown
1#`~astropy.units.Unit` ['angle']#astropy.units.Unit|tuple['angle']
1#list of `astropy.units.Unit`#Sequence[astropy.units.Unit]
1#list of float#Sequence[float]
1#`~matplotlib.figure.Figure`#matplotlib.figure.Figure
1#:class:`~astropy.wcs.WCS`#astropy.wcs.WCS
1#:class:`~astropy.wcs.WCS` or :class:`~matplotlib.transforms.Transform` or str#astropy.wcs.WCS|matplotlib.transforms.Transform|str
1#`.RendererBase` subclass#Unknown
1#list of `.Artist` or ``None``#Sequence[Artist]|None
1#default: False#Unknown
1#'both', 'x', 'y'#Literal['both']|Literal['x']|Literal['y']
1#:class:`~matplotlib.path.Path`#matplotlib.path.Path
1#:class:`~astropy.visualization.wcsaxes.CoordinatesMap`#astropy.visualization.wcsaxes.CoordinatesMap
1#{'longitude', 'latitude', 'scalar'}#Literal['longitude', 'latitude', 'scalar']
1#`~astropy.visualization.wcsaxes.frame.BaseFrame`#astropy.visualization.wcsaxes.frame.BaseFrame
1#{'lines', 'contours'}#Literal['lines', 'contours']
1#str or `~matplotlib.ticker.Formatter`#str|matplotlib.ticker.Formatter
1#{'auto', 'ascii', 'latex'}#Literal['auto', 'ascii', 'latex']
1#str or tuple or None#str|tuple|None
1#class:`~astropy.units.Unit`#class:~astropy.units.Unit
1#{'in','out'}#Literal['in','out']
1#ndarray or a list of arrays#np.ndarray|Unknown
1#`~matplotlib.axes.Axes` instance#matplotlib.axes.Axes
ndarray
str
bool
list
float
ndarray or float
tuple
array-like
`~astropy.units.Quantity` ['length']
`~astropy.units.Quantity`
dict
array
`~astropy.coordinates.Angle`
`~astropy.cosmology.Cosmology` subclass instance
int
float or `~numpy.ndarray`
`~astropy.table.Table`
bytes
int array
np.ndarray
unicode
`~astropy.units.Quantity` ['time']
(2,) tuple
`numpy.ndarray`
`~astropy.io.votable.tree.Element`
callable
`~astropy.units.Quantity` ['angle']
`~astropy.table.Table` object
`BaseCoordinateFrame` subclass instance
float or array
object
`~astropy.coordinates.SkyCoord`
scalar or array
list of str
function
`~astropy.modeling.FittableModel`
`~astropy.wcs.WCS`
`~astropy.coordinates.CartesianRepresentation`
`~astropy.units.Quantity` ['speed']
correctly-typed object, boolean
ndarray or `~astropy.units.Quantity`
array-like, float or None
float or ndarray
`~astropy.units.equivalencies.Equivalency`
list of ndarray
object, or list of object, or list of list of object, or ...
`~astropy.units.CompositeUnit`
int or int array
`~astropy.nddata.NDData`-like
`Header`
`~astropy.coordinates.BaseCoordinateFrame`
ndarray or `~astropy.units.Quantity` ['dimensionless']
float or `~astropy.units.Quantity` ['dimensionless']
source count limit
`~astropy.uncertainty.Distribution` or object
Column
ndarray or scalar
float64
Time object
`~astropy.coordinates.Longitude`
array_like
any type
bool or str
float or `np.ndarray`
.. warning::
1D array
array of dtype float
`~astropy.modeling.core.CompoundModel`
Any
bool or `NotImplemented`
type
`~astropy.table.QTable`
`~astropy.units.Quantity` ['temperature']
scalar or ndarray
iterator
`~astropy.table.Table` if out==None else None
list of `Column`
ndarray, int
:class:`pandas.DataFrame`
Table
`~astropy.units.physical.PhysicalType`
lstr
Latex string
list of `CompositeUnit`
numpy.array, numpy.ma.array
`~datetime.datetime`
Time (or subclass)
float or float array
valid_outputs_unit
float or `~astropy.units.Quantity` ['angle']
int or None
tuple of slice
`NDUncertainty` instance
`NDUncertainty` subclass
`~astropy.io.fits.HDUList`
`~astropy.utils.iers.LeapSeconds`
int, str, str
None
:class:`~astropy.table.Table`
object or None
`~astropy.io.fits.header.Header`
str or None
dict or None
`HDUList`
`astropy.table.Table`
`~astropy.io.ascii.BaseReader` subclass
list of dict
list of `~astropy.table.Table`
`~astropy.coordinates.EarthLocation` (or subclass) instance
`np.ndarray`
bool or array of bool
`SpectralCoord`
str or string array
SkyCoord
`~astropy.coordinates.UnitSphericalRepresentation`
`~astropy.coordinates.Distance`
`SkyCoord`
scalar
array-like or `~astropy.units.Quantity` ['time']
array-like or `~astropy.units.Quantity`
`~astropy.timeseries.TimeSeries`
float or `~astropy.units.Quantity`
float or numpy.ndarray
(N,) array of float
(N-1,) array of float
array of float
`~astropy.units.Quantity` ['dimensionless']
`numpy.array`
`~astropy.cosmology.Parameter`
``validator`` or callable[``validator``]
dict[str, Any]
NotImplemented or True
`math.inf` or ndarray[float] thereof
`~astropy.units.Quantity` ['redshift']
`astropy.table.Table` or subclass instance
`~astropy.table.Row`
`astropy.table.Column`
`astropy.modeling.Parameter`
callable[[`~astropy.io.misc.yaml.AstropyDumper`, |Cosmology|], str]
`_CosmologyModel` subclass instance
explain
`~astropy.units.Quantity` ['frequency']
`~astropy.cosmology.Cosmology` instance
`astropy.cosmology.Cosmology`
np.dtype
Copy of input column
np.ndarray or np.ma.MaskedArray
Column (or subclass)
Column or MaskedColumn
`~astropy.table.Column`
`~astropy.table.MaskedColumn`
None (context manager)
``numpy.void`` or ``numpy.ma.mvoid``
None, tuple
array or `~numpy.ma.MaskedArray`
Column, MaskedColumn, mixin-column type
iterable
Wrapped ``auto_format_func`` function
Table object with groups attr set accordingly
Column object with groups attr set accordingly
None or func
SlicedIndex
Index of columns or None
`Table`
`~astropy.units.PhysicalType`
`~astropy.units.Quantity` (or subclass)
`~astropy.units.Quantity` subclass
`~astropy.units.Quantity`, `~numpy.ndarray`
`astropy.units.format.Base` instance
tuple of lists
tuple of strings
`~astropy.units.UnitBase` instance
`~astropy.units.StructuredUnit`
ndarray view or tuple thereof
`astropy.units.Unit` or tuple
in which case the returned list is always empty.
list of `UnitBase`
`~astropy.units.UnitBase`
Scalar value or numpy array
val1 as dict or None, val2 is always None
`~astropy.time.Time` subclass
list of arrays
:class:`~astropy.time.Time`
`~astropy.time.Time`
str or numpy.array
`~astropy.time.TimeDelta`
ndarray or None
number or ndarray
float or `~astropy.units.Quantity` ['density']
float or `~astropy.units.Quantity` ['speed']
the unique operator key for the dictionary
A validated interval.
Boolean array indicating which parts of _input are outside the interval
An array of the correct shape containing all fill_value
A full set of outputs for case that all inputs are outside domain.
outside the bounding box filled in by fill_value
List of filled in output arrays.
List containing filled in output values and units
bool-numpy array
numpy array
Validated selector_argument
`CompoundModel`
`matplotlib.axes.Axes` instance
new_header, wcs
`dict`
`NDUncertainty` subclass instance or None
`NDData` subclass
`~astropy.nddata.NDData`
readable file-like
file-like
iterator of str
iterator of file object
int or `~astropy.units.Quantity`
IERS
``IERS_A`` class instance
``IERS_B`` class instance
`~astropy.table.QTable` instance
`~collections.OrderedDict` or None
tuple of array
module or None
list of objects
str or bytes
context-dependent
list of unicode
`tuple`
str (one character)
numpy bool array
astropy.io.votable.converters.Converter
`~astropy.io.votable.tree.VOTableFile` object
`~astropy.io.votable.tree.Table` object
`~astropy.io.votable.tree.VOTableFile` instance
writable file-like
``dict_keys``
astropy.time.Time
str or astropy.table.Table
ones complement checksum
ascii encoded checksum
HDUList
BaseHDU
str, number, complex, bool, or ``astropy.io.fits.card.Undefined``
`Header` object
ndarray or `~numpy.recarray` or `~astropy.io.fits.Group`
str, int, or float
`~astropy.io.fits.BinTableHDU`
int, None
(int, int, int, bool, bool)
list of list of str
`~astropy.table.Table` or <generator>
`~astropy.table.Table` or None
Sorted lists
`list` [`str`]
object or list
:class:`~astropy.coordinates.baseframe.BaseCoordinateFrame` subclass instance
:class:`~astropy.wcs.WCS` instance
`~astropy.coordinates.SkyCoord` subclass
`~.builtin_frames.ITRS`
`~astropy.wcs.WCS` object
(4, 2) array of (*x*, *y*) coordinates.
list of `~astropy.units.Quantity`
`astropy.io.fits.Header`
list of `WCS`
list subclass instance
Constant
``configobj.ConfigObj`` or ``configobj.Section``
`BaseRepresentation` or `BaseDifferential` subclass instance
`BaseRepresentation` subclass instance
`CartesianRepresentation`
str or list of str
EarthLocation (or subclass)
astropy.coordinates.sites.SiteRegistry
`numpy.matrix`
`~astropy.coordinates.BaseRepresentation` subclass instance
`~astropy.coordinates.CartesianRepresentation` or tuple
tuple of `~astropy.coordinates.CartesianRepresentation`
`~scipy.spatial.cKDTree` or `~scipy.spatial.KDTree`
Infinite wisdom, condensed into astrologically precise prose.
`~astropy.units.Quantity` ['angle'] or float
`~astropy.coordinates.SphericalRepresentation`
list of class or None
`CompositeTransform` or None
`BaseCoordinateFrame` subclass
dict of subclasses
`~astropy.coordinates.BaseRepresentation` or `~astropy.coordinates.BaseDifferential`.
BaseRepresentation-derived object
coordinate-like
3x3 array
SkyCoord (or subclass)
ndarray, shape=(n_times, n_parameters)
ndarray or `~astropy.units.Quantity` ['frequency']
np.ndarray (n_parameters,)
array-like, `~astropy.units.Quantity`, or `~astropy.time.Time`
`astropy.timeseries.sampled.TimeSeries`
:class:`~astropy.timeseries.BinnedTimeSeries`
:class:`~matplotlib.path.Path`
"""
