# Script to help with occasional grammar testing.
# TODO: move grammar out of herte and get it from docs2stubs.

from lark.lark import Lark
from lark.tree import Tree
from lark.lexer import Token
from lark.visitors import Interpreter


_grammar = """
start: type_list [_PERIOD]
type_list: type ((_COMMA|OR|_COMMA OR) type)*
type: basic_type [TYPE] 
    | alt_type 
    | array_type 
    | restricted_type 
    | tuple_type 
    | class_type 
    | dict_type 
    | list_type 
    | generator_type 
    | union_type 
    | optional_type 
    | literal_type 
    | callable_type 
    | iterable_type
    | filelike_type
    | _LESSTHAN type _GRTRTHAN
alt_type: _LBRACE array_kind (_COMMA array_kind)* _RBRACE [_COMMA] ([shape_qualifier] | [shape]) [type_qualifier]
array_type: [dimension] (arraylike | ndarraylike | basic_type array_type) 
          | shape array_kind [type_qualifier]
          | array_kind [_COMMA] dimension
          | dimension array_kind
          | basic_type array_kind
          | array_kind _LPAREN type _RPAREN
dimension: _DIM ((OR | _SLASH) _DIM)* 
        | (NUMBER|NAME) _X (NUMBER|NAME) 
        | _LPAREN (NUMBER|NAME) _COMMA [NUMBER|NAME] [_COMMA [NUMBER|NAME]] _RPAREN
        | ONED
        | TWOD
        | THREED
ndarraylike: ndarray_kind [shape_qualifier [[_COMMA] type_qualifier]] 
           | ndarray_kind [_COMMA] type_qualifier [[_COMMA] shape_qualifier] 
           | ndarray_kind _LBRACKET basic_type _RBRACKET
array_kind: ndarray_kind 
          | arraylike_kind
ndarray_kind: NDARRAY 
            | MATRIX 
            | ARRAY 
            | ARRAYS 
            | (NDARRAY|NUMPY) [basic_type] ARRAY 
            | basic_type [_DASH] NDARRAY
arraylike: arraylike_kind [ [_COMMA] shape_qualifier [ [_COMMA] type_qualifier]]
arraylike_kind: ARRAYLIKE | ARRAY
shape_qualifier: ([_COMMA] [OF] SHAPE [_EQUALS|OF] shape_specifier (OR shape_specifier)*) 
               | SHAPE _COLON shape_specifier 
               | [_COMMA] [WITH [SHAPE]] [OF] (SIZE|LENGTH) [_LPAREN] (QUALNAME|NUMBER) [_COMMA] [_RPAREN]
               | [_COMMA] [WITH [SHAPE]] [OF] shape_specifier 
               | [_COMMA] [WITH [SHAPE]] [OF] dimension
               | _LPAREN LENGTH (QUALNAME|NUMBER) _RPAREN
               | [_COMMA] SAME SHAPE AS QUALNAME
shape_specifier: shape | NAME (_PERIOD NAME)*
shape: ((_LPAREN|_LBRACKET) (QUALNAME|NUMBER) (_COMMA (QUALNAME|NUMBER))* _COMMA? (_RPAREN|_RBRACKET))
type_qualifier: OF (ARRAYS | array_type | basic_type (OBJECT)?)
              | [OF] DTYPE [_EQUALS] dtype
              | _LBRACKET type _RBRACKET
dtype: dtype_type 
     | _LBRACE dtype_type (_COMMA dtype_type)* [_COMMA] _RBRACE
dtype_type: basic_type [TYPE] | QUALNAME
basic_type: ANY 
            | [POSITIVE|NEGATIVE] INT 
            | STR 
            | [POSITIVE|NEGATIVE] FLOAT [IN range] 
            | BOOL
            | SCALAR [VALUE]
            | COMPLEX
restricted_type: [ONE OF] _LBRACE (literal_type|STR) (_COMMA (literal_type|STR))* _RBRACE
tuple_type: TUPLE 
          | TUPLE (OF|WITH) [NUMBER] type (OR type)*
          | [TUPLE] _LPAREN type (_COMMA type)* _RPAREN
          | [TUPLE] _LBRACKET type (_COMMA type)* _RBRACKET
          | [TUPLE] _LBRACE type (_COMMA type)* _RBRACE
dict_type: (MAPPING|DICT) (OF|FROM) (basic_type|QUALNAME) [TO (basic_type | QUALNAME | ANY)] 
         | DICT _LBRACKET type _COMMA type _RBRACKET
class_type: [CLASSMARKER] [_A|_AN] class_specifier [INSTANCE|OBJECT]
class_specifier: [INSTANCE|SUBCLASS] OF QUALNAME 
               | QUALNAME [_COMMA|_LPAREN] OR [_A] SUBCLASS [OF QUALNAME][_RPAREN]
               | QUALNAME [_COLON QUALNAME] [CLASS|SUBCLASS]
               | QUALNAME _DASH LIKE 
list_type: (LIST|SEQUENCE) _LBRACKET type _RBRACKET
         | [_A] (LIST|SEQUENCE) [OF] type shape_qualifier
         | type SEQUENCE
         | [_A] (LIST|SEQUENCE) OF type (OR type)*
generator_type: GENERATOR [OF type]
iterable_type: ITERABLE [OF type]
range: _LBRACKET NUMBER _COMMA NUMBER _RBRACKET
union_type: UNION _LBRACKET type (_COMMA type)* _RBRACKET | type (AND type)+
optional_type: OPTIONAL [_LBRACKET type _RBRACKET]
literal_type: STRING | NUMBER | NONE | TRUE | FALSE
callable_type: CALLABLE [_LBRACKET _LBRACKET type_list _RBRACKET _COMMA type _RBRACKET]
filelike_type: [READABLE|WRITABLE] FILELIKE [TYPE]


AND.2:       "and"i
ANY.2:       "any"i
ARRAYLIKE.2: "arraylike"i | "array-like"i | "array like"i | "array_like"i | "list"i
ARRAY.2:     "array"i
ARRAYS.2:    "arrays"i
AS.2:        "as"i
BOOL.2:      "bool"i | "bools"i | "boolean"i | "booleans"i
CALLABLE.2:  "callable"i | "function"i
CLASS.2:     "class"i
CLASSMARKER.2:":class:"
COMPLEX.2:   "complex"i
DICT.2:      "dict"i | "dictionary"i
DTYPE.2:     "dtype"i
FALSE.2:     "false"i
FILELIKE.2:  "file-like"i | "filelike"i
FLOAT.2:     "float" | "floats" | "float32"i | "float64"i
FROM.2:      "from"i
GENERATOR.2: "generator"i
IN.2:        "in"i
INSTANCE.2:  "instance"i
INT.2:       "int"| "ints"|  "integer" | "integers" | "int32" | "int64"
ITERABLE.2:  "iterable"i | "iterator"i
LENGTH.2:    "length"i
LIKE.2:      "like"i
LIST.2:      "list"i
MAPPING.2:   "mapping"i
MATRIX.2:    "matrix"i | "sparse matrix"i | "sparse-matrix"i
NDARRAY.2:   "ndarray"i | "nd-array"i | "numpy array"i | "np.array"i
NEGATIVE.2:  "negative"i
NONE.2:      "none"i
NUMPY.2:     "numpy"i
OBJECT.2:    "object"i | "objects"i
OF.2:        "of"i
ONE.2:       "one"i
ONED.2:      "1-d"i | "1d"i
OPTIONAL.2:  "optional"i
OR.2:        "or"i
POSITIVE.2:  "positive"i | "non-negative"i
READABLE.2:  "readable"i | "readonly"i | "read-only"i
SAME.2:      "same"i
SCALAR.2:     "scalar"i
SEQUENCE.2:  "sequence"i
SHAPE.2:     "shape"i
SIZE.2:      "size"i
SORTED.2:    "sorted"i
STR.2:       "str"i | "string"i | "strings"i | "python string"i
SUBCLASS.2:  "subclass"i
THEREOF.2:   "thereof"i
THREED.2:    "3-d"i | "3d"i
TO.2:        "to"i
TRUE.2:      "true"i
TUPLE.2:     "tuple"i | "2-tuple"i | "2 tuple"i | "3-tuple"i | "3 tuple"i | "4-tuple" | "4 tuple"
TWOD.2:      "2-d"i | "2d"i
TYPE.2:      "type"i
UNION.2:     "union"i
VALUE.2:     "value"i
WITH.2:      "with"i
WRITABLE.2:  "writeable"i | "writable"i

_A:         "a"i
_AN:        "an"i
_ASTERISK:  "*"
_BACKTICK:  "`"
_COLON:    ":"
_COMMA:    ","
_DASH:     "-"
_DIM:      "0-d"i | "1-d"i | "2-d"i | "3-d"i | "1d"i | "2d"i | "3d"i
_EQUALS:   "="
_GRTRTHAN:  ">"
_LBRACE:   "{"
_LBRACKET:  "["
_LESSTHAN:  "<"
_LPAREN:    "("
_NEWLINE:   "\n"
_PLURAL:    "\\s"
_PERIOD:   "."
_RBRACE:   "}"
_RBRACKET:  "]"
_RPAREN:    ")"
_SLASH:     "/"
_TILDE:     "~"
_X:         "x"


NAME:      /[A-Za-z_][A-Za-z0-9_\-]*/
NUMBER:    /-?[0-9][0-9\.]*/
QUALNAME:  /\.?[A-Za-z_][A-Za-z_0-9\-]*(\.[A-Za-z_.][A-Za-z0-9_\-]*)*/
STRINGSQ:  /\'[^\']*\'/
STRINGDQ:  /\"[^\"]*\"/
STRING:    STRINGSQ | STRINGDQ

%import common.WS
%ignore WS
%ignore _BACKTICK
%ignore _TILDE
%ignore _ASTERISK
%ignore _PLURAL
%ignore THEREOF
%ignore SORTED

"""


class Normalizer(Interpreter):
    def __init__(self, tlmodule: str, module:str):
        self._tlmodule = module  # top-level module
        self._module = module

    def start(self, tree) -> tuple[str, set[str]|None]:
        """ start: type_list [PERIOD] """
        result = self.visit(tree.children[0])
        return result
        
    def type_list(self, tree) -> tuple[str, set[str]|None]:
        """ type_list: type ((_COMMA|OR|_COMMA OR) type)*  [OR NONE] [_PERIOD] """
        type = ''
        imports = set()
        literals = []
        has_none = False
        for child in tree.children:
            if isinstance(child, Tree):
                result = self._visit_tree(child)
                if result:
                    if result[0] == 'None':
                        has_none = True
                        continue
                    if result[0].startswith('Literal:'):
                        literals.append(result[0][8:])
                    else:
                        if type:
                            type += '| '
                        type += result[0]
                    if result[1]:
                        imports.update(result[1])

        if not imports:
            imports = None
        if literals:
            if type:
                type += '| '
            type += 'Literal[' + ','.join(literals) + ']'            
        if has_none:
            if type:
                type += '| '
            type += 'None'
        return type, imports

    def type(self, tree)-> tuple[str, set[str]|None]:
        """
        type: basic_type [TYPE] 
            | alt_type 
            | array_type 
            | restricted_type 
            | tuple_type 
            | class_type 
            | dict_type 
            | list_type 
            | generator_type 
            | union_type 
            | optional_type 
            | literal_type 
            | callable_type 
            | iterable_type
            | filelike_type
            | _LESSTHAN type _GRTRTHAN
        """
        for child in tree.children:
            print(child)
            if isinstance(child, Tree):
                result = self._visit_tree(child)
                if result:
                    return result
        assert(False)

    _basic_types = {
        'ANY': 'Any',
        'INT': 'int',
        'STR' : 'str',
        'FLOAT': 'float',
        'BOOL': 'bool',
        'SCALAR': 'Scalar',
        'COMPLEX': 'complex'
    }

    def basic_type(self, tree) -> tuple[str, set[str]|None]:
        """
        basic_type: ANY 
            | [POSITIVE|NEGATIVE] INT 
            | STR 
            | [POSITIVE|NEGATIVE] FLOAT [IN range] 
            | BOOL
            | SCALAR [VALUE]
            | COMPLEX
        """
        for child in tree.children:
            if isinstance(child, Token):
                if child.type in self._basic_types:
                    return self._basic_types[child.type], None
                if child.type == 'ANY':
                    return 'Any',set(('typing', 'Any'))
                elif child.type == 'SCALAR':
                    return 'Scalar', set((f'{self._tlmodule}._typing', 'Scalar'))

        assert(False)
    
    def alt_type(self, tree):
        pass

    def array_type(self, tree):
        pass

    def restricted_type(self, tree):
        pass

    def tuple_type(self, tree):
        pass

    def class_type(self, tree):
        pass

    def dict_type(self, tree):
        pass

    def list_type(self, tree):
        pass

    def generator_type(self, tree):
        """ generator_type: GENERATOR [OF type] """
        pass

    def union_type(self, tree):
        pass
    
    def optional_type(self, tree):
        pass

    def literal_type(self, tree)-> tuple[str, set[str]|None]:
        """ literal_type: STRING | NUMBER | NONE | TRUE | FALSE """
        assert(len(tree.children) == 1 and isinstance(tree.children[0], Token))
        tok = tree.children[0]
        type = tok.type
        if type == 'STRING' or type == 'NUMBER':
            return f'Literal:' + tok.value, set(('typing', 'Literal'))
        if type == 'NONE':
            return 'None', None
        if type == 'TRUE':
            return 'Literal:True', set(('typing', 'Literal'))
        if type == 'FALSE':
            return 'Literal:False', set(('typing', 'Literal'))
        assert(False)

    def callable_type(self, tree):
        pass

    def iterable_type(self, tree):
        pass
    
    def filelike_type(self, tree):
        """ filelike_type: [READABLE|WRITABLE] FILELIKE [TYPE] """
        return 'FileLike', set((f'{self._tlmodule}._typing', 'FileLike'))

    
_lark = Lark(_grammar)
_norm = Normalizer('tlmod', 'mod')


def normalize_type(s: str, modname: str|None = None) -> tuple[str, set[str]|None]:
    """ Normalize a type, returning the normalized type and the list of
        required imports, or None if no imports are needed.
    """
    #try:
    if True:
        tree = _lark.parse(s)
        tree.pretty()
        n = _norm.visit(tree)
        return n
    #except Exception as e:
    #    print(e)
    #    return s, None
    
    return s, None


if True:
    lines = ["'true' or 'false' or true or false or 3"]
else:
    with open('testdata.txt') as f:
        lines = [l for l in f]

    
for line in lines:
    l = line.strip()
    result = normalize_type(l)
    print(result)
    print(f'{l} ### {result[0]} ### {result[1]}')

