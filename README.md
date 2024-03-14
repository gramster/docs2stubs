# docs2stubs - Generate Python type stub files for Scientific Python packages

docs2stubs is a utility that can generate Python type stub files (`.pyi` files)
for Scientific Python packages that use numpydoc-format docstrings.

See CONTRIBUTING.md for build instructions, or install from PyPI with:

```
python -m pip install docs2stubs
```

Usage:

```
Usage:
  docs2stubs analyze (package|module) [--include-counts] <name>...
  docs2stubs stub (package|module) [--strip-defaults] <name>...
  docs2stubs test [<name>] <typestring>
  docs2stubs -h | --help
  docs2stubs --version

Options:
  --strip-defaults  Replace parameter default values with ...
  <name>            The target package or module (e.g. matplotlib.pyplot).
```

The package/module needs to be installed in the environment from which
you are running docs2stubs.

The `analyze` command parses numpydoc-format docstrings in the target
and produces files 'analysis/\<name\>.\<what\>.map.missing' which lists the
non-trivial discovered types in order of frequency, along with a normalized version
(separated by ':'). The normalized version is a 'best-effort' corrected
version of the type suitable for re-insertion in type stubs. The fields
are separated by '#'. The analysis phase also outputs a `.imports.map`
file enumerating the discovered classes and their modules; this file can
be augmented with additional class names and owner modules which will 
be used when stubbing to determine what imports might need to be added.

Separate files are generated for each of parameters, return types, or
attributes (the \<what>\ in the filenames above).

If `--include-counts` is specified, then the output files include the
frequency counts in the first field.

After analysis a 'human in the loop' pass of the .map file is 
recommended to do further cleanup of the normalized forms. Once the 
map file is cleaned up, it can be renamed '.map' instead of 
'.map.missing', and will be used during stub generation to supplement
the logic of the type normalizer.

The `analyze` command just does the first phase of gathering types
and writing the `.map.missing` files. It will output a summary 
at the end of how many of the docstring types were found in the 
`.map` files (if present), how many were trivially convertible 
to type annotations, and how many were unhandled ('missed') and 
thus written to the `map.missing` files. Note that the map files
contain just one entry per type, while a type can occur multiple
times, so the output summary counts are typically much higher 
than the count of lines in the mapfiles.

The `stub` command will generate stubs for the module/package under a 
'typings' folder. It will first run an analysis pass and
then make use of the .map file produced by analysis to 
determine types, and for assignments or default parameter values with
no normalized type available, will infer the type from the assigned
value if possible.

Parameter default values are usually preserved in the stubs if they are
scalars, but if you specifiy `--strip-defaults` they will all be removed
and replaced with '...'.

The `test` command can be used to test how a type string will be 
normalized by the program. A module or package name can optionally
be specified in which case the `.imports.map` file for that module
or package will be used (if it exists) to decide what are valid classnames.


## Adding a Supplementary Typing Module

There are some types that don't exist in Python typing that are used 
in several Scientific Python libraries. To support these, you should 
add a module to the target package that defines them as you want them.
For example, for `matplotlib` I added a `_typing.py` file:

```python
import decimal
import io
import numpy.typing
import pandas as pd

Decimal = decimal.Decimal
PythonScalar = str | int | float | bool

ArrayLike = numpy.typing.ArrayLike
FileLike = io.IOBase
PathLike = str

PandasScalar = pd.Period | pd.Timestamp | pd.Timedelta | pd.Interval
Scalar = PythonScalar | PandasScalar

Color = tuple[float, float, float] | str

__all__ = [
    "ArrayLike",
    "Color",
    "Decimal",
    "FileLike",
    "PathLike",
    "Scalar",
]
```

`docs2stubs` is aware of some of these (namely `Scalar`, `ArrayLike`, `FileLike`, and `PathLike`) and will normalize types
like `array-like` to the corresponding type.

## TODO

- Currently the analysis phase handles the Attributes section of a docstring,
but the stub phase does not.
- Would be good to format the stub files with black
- Import statements are added as needed but using absolute imports; releative
imports should be an option.
- Little effort is made to remove 'private' functions/methods/classes.


## Version History

0.1 - Initial release

