import re
from .utils import load_map

"""
TYPEDOC = TYPEPREFIX | TYPEPREFIX or None
TYPEPREFIX = TYPE | TYPE or TYPEPREFIX | TYPE, TYPEPREFIX
TYPE = ALTTYPE | ARRAYLIKE | BASICTYPE | RESTRICTED
ALTTYPE='{' ARRAYKIND [',' ARRAYKIND]* '}' ARRAYQUALIFIER


ARRAYLIKE=ARRAYKIND ARRAYQUALIFIER
ARRAYKIND = (ndarray|matrix|sparse matrix) 
ARRAYQUALIFIER = SHAPEWUALIFIER | TYPEQUALIFIER
SHAPEQUALIFIER = of shape '(' <id> [',' <id>]* ')'
TYPEQUALIFIER = of BASICTYPE
BASICTYPE=int|str|float|
RESTRICTED = '{' STRING [',' STRING]* '}'
"""

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
_shaped = re.compile(r'^(.*)( of shape |, shape )\([^\)]*\)( or \([^\)]*\))*$')

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

  'ndarray': 'np.ndarray',
  'sparse matrix': 'np.ndarray',
}


def is_trivial(s, modname: str, classes: set|dict|None = None):
    """
    s - the type docstring to check
    modname - the module name
    classes - a set of class names or dictionary keyed on classnames 
    """
    s = s.lower()
    if s.endswith(' or none'):
        s = s[:-7]

    if m:=_shaped.match(s):
        s = m.group(1)

    if s.find(' or ') > 0:
        if all([is_trivial(c.strip(), modname, classes) for c in s.split(' or ')]):
            return True

    if _single_restricted.match(s):
        return True
        
    if _ndarray.match(s) or _arraylike.match(s):
        return True

    for r in [_sequence_of, _set_of, _tuple_of]:
        match = r.match(s)
        if match:
            return is_trivial(match.group(2), modname, classes)

    match = _dict_of.match(s)
    if match:
        return is_trivial(match.group(2), modname, classes) and \
            is_trivial(match.group(3), modname, classes)

    nt = normalize_type(s)

    if nt.lower() in _basic_types:
        return True

    if classes:
        # Check unqualified classname

        if nt in classes: # 
            return True

        # Check full qualified classname
        if _ident.match(nt) and nt.startswith(modname + '.'):
            return True
            #if nt[nt.rfind('.')+1:] in classes:
            #    return True

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


def normalize_type(s: str, modname: str|None = None) -> str:

    s = s.strip()
    sl = s.lower()

    # A lot of types end with 'or None', so let's handle
    # that first to simplify everything else.
   
    ornone = ''  # We'll append this before returning
    if sl.endswith(' or none'):
        s = s[:-7]
        sl = sl[:-7]
        ornone = '|None'

    if m:=_shaped.match(s):
        s = m.group(1)

    if _ndarray.match(sl):
        s = 'np.ndarray'
    elif _arraylike.match(sl):
        s = 'ArrayLike'

    def recombine(t1, t2, t3) -> str:
        t1 = normalize_type(t1)
        if t1:
            t1 += '|'
        t3 = normalize_type(t3)
        if t3:
            t3 += '|'
        return t1 + t2 + t3

    # Handle a restricted value set
    m = _restricted_val.match(s)
    if m:
        # Make sure the innards are a comma-separated list
        # of strings
        opts = m.group(2).split(',')
        if all([_is_string(opt.strip()) for opt in opts]):
            return recombine(m.group(1), 'Literal[' + m.group(2) + ']', m.group(3)) + ornone
        elif not m.group(1): # scikit uses this for alternates "{a, b} of x"
            s = ' or '.join([(o + m.group(3)) for o in opts])

    # Handle tuples in [] or (). Right now we can only handle one per line;
    # need to fix that.

    m = _tuple1.match(s)
    if not m:
        m = _tuple2.match(s)
    if m:
        return recombine(m.group(1), 'tuple[' + m.group(2) + ']', m.group(3)) + ornone

    # Now look at list of types. First replace ' or ' with a comma.
    # This is a bit dangerous as commas may exist elsewhere but 
    # until we find the failure cases we don't know how to address 
    # them yet.
    s = s.replace(' or ', ',')

    # Get the alternatives
    parts = s.split(',')

    def normalize_one(s: str, modname: str|None = None):

        orig = s
        rtn = None
        """ Do some normalizing of a single type. """
        s = s.strip()
        if s.startswith(':class:'):
            s = s[7:].strip()

        if s.endswith(' instance'):
            s = s[:-9].strip()

        s = s.replace('`', '')  # Removed restructured text junk

        # Handle collections like 'list of...', 'array of ...' ,etc
        if m:=_sequence_of.match(s):
            rtn = f'Sequence[{normalize_one(m.group(2))}]'
        elif m:=_set_of.match(s):
            rtn = f'set[{normalize_one(m.group(2))}]'
        elif m:=_tuple_of.match(s):
            rtn = f'tuple[{normalize_one(m.group(2))}, ...]'
        elif m:=_dict_of.match(s):
            rtn = f'set[{normalize_one(m.group(2))}, {normalize_one(m.group(3))}]'
        elif _filelike.match(s):
            rtn = 'FileLike'
        elif _pathlike.match(s):
            rtn = 'PathLike'
        elif s.lower() == 'scalar':
            rtn = 'Scalar'
        elif _is_string(s):
            rtn = 'Literal[' + s + ']'
        else:
            try:
                float(s)
                rtn = 'Literal[' + s + ']'
            except ValueError:
                while s.startswith('.') or s.startswith('~'):
                    s = s[1:]

                if s in _basic_types:
                    rtn = _basic_types[s]

                elif modname and s.startswith(modname + '.') and _ident.match(s):
                    rtn = s[s.rfind('.')+1:]
                elif s.find(' ') >= 0:
                    rtn = 'Unknown'
                else:
                    rtn = s
        
        if rtn:
            _norm1[orig] = rtn
        return rtn
        
    # Create a union from the normalized alternatives
    s = '|'.join(normalize_one(p) for p in parts if p.strip())

    return s + ornone


def check_normalizer(typ: str, m: str|None=None):
    classes = set()
    if m:
        classes = load_map(f'analysis/{m}.imports.map')
    else:
        m = ''
    trivial = is_trivial(typ, m, classes)
    normalized = normalize_type(typ, m)
    return f"{'(Trivial)' if trivial else '(Non-Trivial)'} {normalized}"
