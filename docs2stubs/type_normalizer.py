import re
from .type_parser import parse_type
from .utils import load_map


_ident = re.compile(r'^[A-Za-z_][A-Za-z_0-9\.]*$')
_restricted_val = re.compile(r'^(.*){(.*)}(.*)$')
_tuple1 = re.compile(r'^(.*)\((.*)\)(.*)$')  # using ()
_tuple2 = re.compile(r'^(.*)\[(.*)\](.*)$')  # using []
_sequence_of = re.compile(r'^(List|list|Sequence|sequence|Array|array) of ([A-Za-z0-9\._~`]+)$')
_set_of = re.compile(r'^(Set|set) of ([A-Za-z0-9\._~`]+)$')
_tuple_of = re.compile(r'^(Tuple|tuple) of ([A-Za-z0-9\._~`]+)$')
_dict_of = re.compile(r'^(Dict|dict) of ([A-Za-z0-9\._~`]+) to ([A-Za-z0-9\._~`]+)$')
_ndarray = re.compile(r'^ndarray(( of|,) (shape|[a-z]+)[ ]*\([^)]*\))?(, dtype=[a-z]+)?$')
_arraylike = re.compile(r'^array(\-)?(like)? ?(( of|,) shape[ ]*\([^)]*\))?$')
_filelike = re.compile(r'^file(-)?like$')
_pathlike = re.compile(r'^path(-)?like$')
_shaped = re.compile(r'^(.*)( shape)([ =][\[\(])([^\]\)]*)([\]\)])([ ]*or[ ]*\([^\)]*\))*(.*)$', flags=re.IGNORECASE)

# Start with {, end with }, comma-separated quoted words
_single_restricted = re.compile(r'^{([ ]*[\"\'][A-Za-z0-9\-_]+[\"\'][,]?)+}$') 


_basic_types = {
    # Key: lower() version of type
    'any': 'Any', 
    'array': 'NDArray',
    'arraylike': 'NDArray',
    'array-like': 'NDArray',
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
    'ndarray': 'np.ndarray',
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


def remove_shape(s: str) -> str:
    if m:=_shaped.match(s):
        # Drop group 6 and everything except commas from group 4
        # This will change:
        #    {ndarray, sparse matrix} of shape (n_samples, n_classes) or (M, N)
        # to:
        #    {ndarray, sparse matrix} of shape (,)
        first_shape = m.group(4)
        parts = first_shape.split(',')
        first_shape = ','.join('N' * len(parts))
        return m.group(1) + m.group(2) + ' (' + first_shape + ')' + m.group(7)
    else:
        return s


def is_trivial(s, modname: str, classes: set|dict|None = None):
    """
    Returns true if the docstring is trivially and unambiguously convertible to a 
    type annotation, and thus need not be written to the map file for further
    tweaking.

    TODO: this is probably too generous and matplotlib-specific. I think it needs to
    be tightened to a smaller set. Because of that I am temporarily inserting a 
    return False at the start and will then revisit the rest.

    s - the type docstring to check
    modname - the module name
    classes - a set of class names or dictionary keyed on classnames 
    """
    s = s.strip()
    sl = remove_shape(s.lower())
    if sl.endswith(" objects"):
        sl = sl[:-8]

    if sl in _basic_types:
        return True

    # Handle things of form "basic_type or ..."
    #parts = sl.split(' or ') 
    #if parts[0] in _basic_types:
    #    return is_trivial(' or '.join(parts[1:]), modname, classes)  

    # These are low frequency enough they're not worth the effort to handle
    #if sl.startswith("generator of "):
    #    return is_trivial(s[13:], modname, classes) 
    #if sl.startswith("iterable over "):
    #    return is_trivial(s[14:], modname, classes) 

    # Check if it's a string

    if sl and sl[0] == sl[-1] and (sl[0] == '"' or sl[0] =="'"):
        return True

    if classes:
        if s in classes or (_ident.match(s) and s.startswith(modname + '.')):
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
    if sl.startswith(':class:`~') and sl.endswith('`') and is_trivial(sl[9:-1], modname, classes):
        return True
    # Foo instance/instance of Foo
    if sl.endswith(' instance') and is_trivial(sl[:-9], modname, classes):
        return True
    if sl.startswith('instance of ') and is_trivial(sl[12:], modname, classes):
        return True
    if sl.startswith('matplotlib '):
        return is_trivial(sl[11:], modname, classes)
    
    # Handle tuples
    if sl.startswith('tuple'):
        sx = s[5:].strip()
        if not sx:
            return True
        if sx[0] in '({[' and sx[-1] in '})]':
            # TODO We should make sure there are no other occurences of these
            # A lot of this is getting to where we should go back to regexps.
            return is_trivial(sx[1:-1], modname, classes)
        
        # Strip off leading OF or WITH
        if sx.startswith ('of ') or sx.startswith('with '):
            x = sx.find(' ')
            sx = sx[x+1:].strip()
        
        # Strip off a number
        if sx and sx[0].isdigit():
            x = sx.find(' ')
            sx = sx[x+1:]

        if is_trivial(sx, modname, classes):
            return True
        
    for s1 in [s, s.replace(',', ' or ')]:
        for splitter in [' or ', '|']:
            if s1.find(splitter) > 0:
                if all([len(c.strip())==0 or is_trivial(c.strip(), modname, classes) \
                        for c in s1.split(splitter)]):
                    return True
        
    if s.find(' of ') > 0:
        # Things like sequence of int, set of str, etc
        parts = s.split(' of ')
        if len(parts) == 2 and is_trivial(parts[0], modname, None) and is_trivial(parts[1], modname, classes):
            return True
        
    # Handle restricted values in {}
    if s.startswith('{') and s.endswith('}'):
        parts = s[1:-1].split(',')
        parts = [p.strip() for p in parts]
        return all([is_trivial(p, modname, None) for p in parts])
    
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


def normalize_type(s: str, modname: str|None = None, classes: dict|None = None, is_param: bool = False) -> tuple[str|None, dict[str, list[str]]]:
    try:
        return parse_type(remove_shape(s), modname, classes, is_param)
    except Exception as e:
        return None, {}
    


def check_normalizer(typ: str, m: str|None=None, classes: dict|None = None):
    if m is None:
        m = ''
    if classes is None:
        classes = {}
        if m:
            classes = load_map(m, 'imports')

    trivial = is_trivial(typ, m, classes)
    normalized = normalize_type(typ, m, classes)

    return trivial, normalized[0], normalized[1]
