import re
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

# Start with {, end with }, comma-separated quoted words
_single_restricted = re.compile(r'^{([ ]*[\"\'][A-Za-z0-9\-_]+[\"\'][,]?)+}$') 


_basic_types = {
  # Key: lower() version of type
  'none': 'None',
  'float': 'float',
  'floats': 'float',
  'int': 'int',
  'ints': 'int',
  'complex': 'complex',
  'bool': 'bool',
  'bools': 'bool',
  'boolean': 'bool',
  'booleans': 'bool',
  'str': 'str',
  'string': 'str',
  'strings': 'str',
  'set': 'set',
  'frozenset': 'frozenset',
  'range': 'range',
  'bytearray': 'bytearray',
  'memoryview': 'memoryview',
  'list': 'list',
  'dict': 'dict',
  'dictionary': 'dict',
  'dictionaries': 'dict',
  'tuple': 'tuple',
  'tuples': 'tuple',
  'object': 'Any',
  'objects': 'Any',
  'any': 'Any', 
  'callable': 'Callable',
  'iterable': 'Iterable',
  'scalar': 'Scalar',
  'arraylike': 'ArrayLike',
  'filelike': 'FileLike',
  'pathlike': 'PathLike',
  'sequence': 'Sequence',
}


def is_trivial(s, m: str, classes: set|dict|None = None):
    """
    s - the type docstring to check
    m - the module name
    classes - a set of class names or dictionary keyed on classnames 
    """
    s = s.lower()
    if s.endswith(' or none'):
        s = s[:-7]

    if s.find(' or ') > 0:
        if all([is_trivial(c.strip(), m, classes) for c in s.split(' or ')]):
            return True

    if _single_restricted.match(s) or _ndarray.match(s) or _arraylike.match(s):
        return True

    for r in [_sequence_of, _set_of, _tuple_of]:
        match = r.match(s)
        if match:
            return is_trivial(match.group(2), m, classes)

    match = _dict_of.match(s)
    if match:
        return is_trivial(match.group(2), m, classes) and \
            is_trivial(match.group(3), m, classes)

    nt = normalize_type(s)

    if nt.lower() in _basic_types:
        return True

    if classes:
        # Check unqualified classname

        if nt in classes: # 
            return True

        # Check full qualified classname
        if _ident.match(nt) and nt.startswith(m + '.'):
            return True
            #if nt[nt.rfind('.')+1:] in classes:
            #    return True

    return False


def normalize_type(s: str) -> str:
    ornone = ''
    s = s.strip()
    if s.endswith(' or none'):
        s = s[:-7]
        ornone = '|None'

    sl = s.lower()
    if _ndarray.match(sl):
        s = 'np.ndarray'
    elif _arraylike.match(sl):
        s = 'ArrayLike'

    # Handle a restricted value set
    m = _restricted_val.match(s)
    l = None
    if m:
        s = m.group(1) + m.group(3)
        l = 'Literal[' + m.group(2) + ']'

    # Handle tuples in [] or (). Right now we can only handle one per line;
    # need to fix that.

    m = _tuple1.match(s)
    if not m:
        m = _tuple2.match(s)
    t = None
    if m:
        s = m.group(1) + m.group(3)
        t = 'tuple[' + m.group(2) + ']'

    # Now look at list of types. First replace ' or ' with a comma.
    # This is a bit dangerous as commas may exist elsewhere but 
    # until we find the failure cases we don't know how to address 
    # them yet.
    s = s.replace(' or ', ',')

    # Get the alternatives
    parts = s.split(',')

    def normalize_one(s: str):

        """ Do some normalizing of a single type. """
        s = s.strip()
        if s.startswith(':class:'):
            s = s[7:].strip()

        s = s.replace('`', '')  # Removed restructured text junk

        # Handle collections like 'list of...', 'array of ...' ,etc
        m = _sequence_of.match(s)
        if m:
            return f'Sequence[{normalize_one(m.group(2))}]'
        m = _set_of.match(s)
        if m:
            return f'set[{normalize_one(m.group(2))}]'
        m = _tuple_of.match(s)
        if m:
            return f'tuple[{normalize_one(m.group(2))}, ...]'
        m = _dict_of.match(s)
        if m:
            return f'set[{normalize_one(m.group(2))}, {normalize_one(m.group(3))}]'

        if _filelike.match(s):
            return 'FileLike'
        elif _pathlike.match(s):
            return 'PathLike'
        elif s.lower() == 'scalar':
            return 'Scalar'
        # Handle literal numbers and strings
        if not (s.startswith('"') or s.startswith("'")):
            try:
                float(s)
            except ValueError:
                while s.startswith('.') or s.startswith('~'):
                    s = s[1:]

                if s in _basic_types:
                    return _basic_types[s]

                return s
        return 'Literal[' + s + ']'
        
    # Create a union from the normalized alternatives
    s = '|'.join(normalize_one(p) for p in parts if p.strip())

    # Add back our constrained value literal, if it exists
    if s and l:
        s += '|' + l
    elif l:
        s = l

    # Add back our tuple, if it exists
    if s and t:
        s += '|' + t
    elif t:
        s = t

    return s + ornone


def check_normalizer(typ: str, m: str|None=None):
    classes = set()
    if m:
        classes = load_map(f'analysis/{m}.imports.map')
    else:
        m = ''
    trivial = is_trivial(typ, m, classes)
    normalized = normalize_type(typ)
    return f"{'(Trivial)' if trivial else '(Non-Trivial)'} {normalized}"
