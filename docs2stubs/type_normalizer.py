from functools import lru_cache
import re
from .type_parser import parse_type
from .utils import load_map


_ident = re.compile(r'^[A-Za-z_][A-Za-z_0-9\.]*$')
_shaped = re.compile(r'^(.*)( shape)([ =][\[\(])([^\]\)]*)([\]\)])([ ]*or[ ]*\(([^\)]*)\))*(.*)$', flags=re.IGNORECASE)

# Start with {, end with }, comma-separated quoted words
#_single_restricted = re.compile(r'^{([ ]*[\"\'][A-Za-z0-9\-_]+[\"\'][,]?)+}$') 


_basic_types = {
    # Key: lower() version of type
    'any': 'Any', 
    'array': 'numpy.typing.NDArray',
    'arraylike': 'numpy.typing.NDArray',
    'array-like': 'numpy.typing.NDArray',
    'bool': 'bool',
    'bools': 'bool',
    'boolean': 'bool',
    'booleans': 'bool',
    'bytearray': 'bytearray',
    'callable': 'Callable',
    'complex': 'complex',
    'dict': 'dict',
    'dictionary': 'dict',
    'dictionaries': 'dict',
    'filelike': 'FileLike',
    'file-like': 'FileLike',
    'float': 'float',
    'floats': 'float',
    'frozenset': 'frozenset',
    'int': 'int',
    'ints': 'int',
    'iterable': 'Iterable',
    'list': 'list',
    'memoryview': 'memoryview',
    'ndarray': 'numpy.ndarray',
    'none': 'None',
    'object': 'Any',
    'objects': 'Any',
    'pathlike': 'PathLike',
    'path-like': 'PathLike',
    'range': 'range',
    'scalar': 'Scalar',
    'sequence': 'Sequence',
    'set': 'set',
    'str': 'str',
    'string': 'str',
    'strings': 'str',
    'tuple': 'tuple',
    'tuples': 'tuple',
}


def sanitize_shape(s: str) -> str:
    if m:=_shaped.match(s):
        # Drop group 6 and everything except commas from group 4
        # This will change:
        #    {ndarray, sparse matrix} of shape (n_samples, n_classes) or (M, N)
        # to:
        #    {ndarray, sparse matrix} of shape (N,N) @ (N,N)
        #
        # This is easier for the parser to handle.
        first_shape = ','.join(['N' if p else '' for p in m.group(4).split(',')])
        if m.group(7):
            second_shape = ','.join(['N' if p else '' for p in m.group(7).split(',')])
            return m.group(1) + m.group(2) + ' (' + first_shape + ') @ (' + second_shape + ')' + m.group(8)
        return m.group(1) + m.group(2) + ' (' + first_shape + ')' + m.group(8)
    else:
        return s


_trivial_cache = {}


def is_trivial(s, modname: str):
    key = f'{modname}.{s}'
    if key in _trivial_cache:
        return _trivial_cache[key]
    rtn = _is_trivial(s, modname)
    _trivial_cache[key] = rtn
    return rtn


def _is_trivial(s, modname: str):
    """
    Returns true if the docstring is trivially and unambiguously convertible to a 
    type annotation, and thus need not be written to the map file for further
    tweaking.

    s - the type docstring to check
    modname - the module name
    """
    s = s.strip()
    sl = sanitize_shape(s.lower())
    if sl.endswith(" objects"):
        sl = sl[:-8]

    if sl in _basic_types:
        return True

    # Check if it's a string

    if sl and sl[0] == sl[-1] and (sl[0] == '"' or sl[0] =="'"):
        return True

    if _ident.match(s) and s.startswith(modname + '.'):
        return True

    # We have to watch out for ambiguous things like 'list of str or bool'.
    # This is just a kludge to look for both 'of' and 'or' in the type and
    # reject it.
    x1 = sl.find(' of ')
    x2 = sl.find(' or ')
    if x1 >= 0 and x2 > x1:
        return False
    
    # Handle list/ArrayLike/NDArray of shape (...)
    # First strip off ", dtype..."" if present
    x = sl.find(', dtype')
    sltmp = sl
    if x >= 0 and sl[x+6:].find(' ') < 0 and sl[x+6:].find(',') < 0:
        sltmp = sl[:x]
    parts = sltmp.split(' of shape ')
    if len(parts) == 2 and parts[0] in ['array', 'arraylike', 'array-like', 'list', 'ndarray'] \
            and parts[1].startswith('(') and parts[1].endswith(')'):
        return True

    # Handle some class cases
    # :class:`~sklearn.preprocessing.LabelEncoder`
    if sl.startswith(':class:`~') and sl.endswith('`') and is_trivial(sl[9:-1], modname):
        return True
    # Foo instance/instance of Foo
    if sl.endswith(' instance') and is_trivial(sl[:-9], modname):
        return True
    if sl.startswith('instance of ') and is_trivial(sl[12:], modname):
        return True
    if sl.startswith('matplotlib '):
        return is_trivial(sl[11:], modname)
    
    # Handle tuples
    if sl.startswith('tuple'):
        sx = s[5:].strip()
        if not sx:
            return True
        if sx[0] in '({[' and sx[-1] in '})]':
            # TODO We should make sure there are no other occurences of these
            # A lot of this is getting to where we should go back to regexps.
            return is_trivial(sx[1:-1], modname)
        
        # Strip off leading OF or WITH
        if sx.startswith ('of ') or sx.startswith('with '):
            x = sx.find(' ')
            sx = sx[x+1:].strip()
        
        # Strip off a number
        if sx and sx[0].isdigit():
            x = sx.find(' ')
            sx = sx[x+1:]

        if is_trivial(sx, modname):
            return True
        
    for s1 in [s, s.replace(',', ' or ')]:
        for splitter in [' or ', '|']:
            if s1.find(splitter) > 0:
                if all([len(c.strip())==0 or is_trivial(c.strip(), modname) \
                        for c in s1.split(splitter)]):
                    return True
        
    if s.find(' of ') > 0:
        # Things like sequence of int, set of str, etc
        parts = s.split(' of ')
        if len(parts) == 2 and is_trivial(parts[0], modname) and is_trivial(parts[1], modname):
            return True
        
    # Handle restricted values in {}
    if s.startswith('{') and s.endswith('}'):
        parts = s[1:-1].split(',')
        parts = [p.strip() for p in parts]
        return all([is_trivial(p, modname) for p in parts])
    
    return False


_norm1 = {}


def print_norm1():
    print("\n\nNORM1\n=====\n")
    for k, v in _norm1.items():
        print(f"{k} # {v}")


def _is_string(s) -> bool:
    # Must start and end with a quote
    if len(s) < 2 or (s[0] != '"' and s[0] != "'"):
        return False 
    # Must start/end with same char and have no intervening 
    # occurences of that char. We don't handle escaped quotes.
    if s[-1] != s[0] or s[1:].find(s[0]) != len(s)-2:
        return False
    return True


_normalize_cache = {}


def normalize_type(s: str, modname: str|None = None, is_param: bool = False) -> str|None:
    try:
        key = f'{modname}.{s}?{is_param}'
        if key in _normalize_cache:
            return _normalize_cache[key]
        rtn = parse_type(sanitize_shape(s), modname, is_param)
        _normalize_cache[key] = rtn
        return rtn
    except Exception as e:
        return None
    

def check_normalizer(typ: str, is_param: bool, m: str|None=None):
    if m is None:
        m = ''

    trivial = is_trivial(typ, m)
    normalized = normalize_type(typ, m, is_param)

    return trivial, normalized
