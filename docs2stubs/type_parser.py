from lark.lark import Lark
from lark.tree import Tree
from lark.lexer import Token
from lark.visitors import Interpreter


_grammar = r"""
start: type_list [_PERIOD]
type_list: type ((_COMMA|OR|_COMMA OR) type)*
type: alt_type 
    | array_type 
    | basic_type [TYPE]
    | callable_type 
    | class_type 
    | dict_type 
    | filelike_type
    | generator_type 
    | iterable_type
    | literal_type 
    | optional_type 
    | restricted_type 
    | set_type
    | tuple_type 
    | union_type 
    | _LESSTHAN type _GRTRTHAN
alt_type: _LBRACE array_kind (_COMMA array_kind)* _RBRACE [_COMMA] [shape_qualifier] [type_qualifier]
array_type: [NDARRAY|NUMPY] basic_type [_DASH] array_kind [[_COMMA] (dimension | shape_qualifier)]
          | array_kind [_COMMA] shape_qualifier [[_COMMA] type_qualifier] 
          | array_kind [_COMMA] type_qualifier [[_COMMA] shape_qualifier] 
          | (dimension | shape) array_kind [type_qualifier]
          | array_kind
array_kind: ARRAYLIKE 
          | LIST
          | NDARRAY 
          | MATRIX 
          | SEQUENCE
          | ARRAY 
          | ARRAYS 
dimension: _DIM ((OR | _SLASH) _DIM)* 
        | (NUMBER|NAME) _X (NUMBER|NAME) 
        | _LPAREN (NUMBER|NAME) _COMMA [NUMBER|NAME] [_COMMA [NUMBER|NAME]] _RPAREN
        | ONED
        | TWOD
        | THREED
shape_qualifier: [[WITH|OF] SHAPE] [_EQUALS|OF] (SIZE|LENGTH) (QUALNAME|NUMBER|shape)
               | [[WITH|OF] SHAPE] [_EQUALS|OF] shape (OR shape)* [dimension]
               | SAME SHAPE AS QUALNAME
shape: (_LPAREN|_LBRACKET) (QUALNAME|NUMBER) (_COMMA (QUALNAME|NUMBER))* _COMMA? (_RPAREN|_RBRACKET)
type_qualifier: OF ARRAYS
              | OF type 
              | [OF] DTYPE [_EQUALS] (basic_type | QUALNAME) [TYPE]
              | _LBRACKET type _RBRACKET
              | _LPAREN type _RPAREN
basic_type: ANY 
            | [POSITIVE|NEGATIVE] INT 
            | STR 
            | [POSITIVE|NEGATIVE] FLOAT [IN _LBRACKET NUMBER _COMMA NUMBER _RBRACKET] 
            | BOOL
            | SCALAR [VALUE]
            | COMPLEX
            | OBJECT
            | FILELIKE
            | PATHLIKE
callable_type: CALLABLE [_LBRACKET _LBRACKET type_list _RBRACKET _COMMA type _RBRACKET]
class_type: [CLASSMARKER] class_specifier [INSTANCE|OBJECT]
        | class_specifier [_COMMA|_LPAREN] OR [_A] SUBCLASS [_RPAREN]
        | class_specifier [_COMMA|_LPAREN] OR class_specifier[_RPAREN]
class_specifier: [_A|_AN] (INSTANCE|CLASS|SUBCLASS) OF QUALNAME 
        | [_A|_AN] QUALNAME (INSTANCE|CLASS|SUBCLASS)
        | [_A|_AN] QUALNAME [_COMMA|_LPAREN] OR [_A|_AN] SUBCLASS [OF QUALNAME][_RPAREN]
        | [_A|_AN] QUALNAME [_COLON QUALNAME] [_DASH LIKE]
dict_type: (MAPPING|DICT) (OF|FROM) (basic_type|QUALNAME) [TO (basic_type | QUALNAME | ANY)] 
         | (MAPPING|DICT) [_LBRACKET type _COMMA type _RBRACKET]
filelike_type: [READABLE|WRITABLE] FILELIKE [TYPE]
generator_type: GENERATOR [OF type]
iterable_type: ITERABLE [OF type]
literal_type: STRING | NUMBER | NONE | TRUE | FALSE
optional_type: OPTIONAL [_LBRACKET type _RBRACKET]
restricted_type: [ONE OF] _LBRACE (literal_type|STR) (_COMMA (literal_type|STR))* _RBRACE
set_type: (FROZENSET|SET) _LBRACKET type _RBRACKET
         | [_A] (FROZENSET|SET) [OF type_list]
tuple_type: TUPLE 
          | TUPLE (OF|WITH) [NUMBER] type (OR type)*
          | [TUPLE] _LPAREN type (_COMMA type)* _RPAREN
          | [TUPLE] _LBRACKET type (_COMMA type)* _RBRACKET
          | [TUPLE] _LBRACE type (_COMMA type)* _RBRACE
union_type: UNION _LBRACKET type (_COMMA type)* _RBRACKET | type (AND type)+
          | type (_PIPE type)*


AND.2:       "and"i
ANY.2:       "any"i
ARRAYLIKE.2: "arraylike"i | "array-like"i | "array like"i | "array_like"i
ARRAY.2:     "array"i
ARRAYS.2:    "arrays"i
AS.2:        "as"i
BOOL.2:      "bool"i | "bools"i | "boolean"i | "booleans"i
CALLABLE.2:  "callable"i | "function"i
CLASS.2:     "class"i
CLASSMARKER.2:":class:"
COMPLEX.2:   "complex"i
DICT.2:      "dict"i | "dictionary"i | "dictionaries"i
DTYPE.2:     "dtype"i
FALSE.2:     "false"i
FILELIKE.2:  "file-like"i | "filelike"i
FLOAT.2:     "float" | "floats" | "float32"i | "float64"i
FROM.2:      "from"i
FROZENSET.2: "frozenset"i
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
PATHLIKE.2:  "path-like"i | "pathlike"i
POSITIVE.2:  "positive"i | "non-negative"i
READABLE.2:  "readable"i | "readonly"i | "read-only"i
SAME.2:      "same"i
SCALAR.2:     "scalar"i
SEQUENCE.2:  "sequence"i
SET.2:       "set"i
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
_PIPE:      "|"
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

    def handle_qualname(self, name: str, imports: set) -> str:
        return name
    
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
                            type += '|'
                        type += result[0]
                    if result[1]:
                        imports.update(result[1])

        if not imports:
            imports = None
        if literals:
            if type:
                type += '|'
            type += 'Literal[' + ','.join(literals) + ']'            
        if has_none:
            if type:
                type += '|'
            type += 'None'
        return type, imports

    def type(self, tree)-> tuple[str, set[str]|None]:
        """
        type: alt_type 
            | array_type 
            | basic_type [TYPE]
            | callable_type 
            | class_type 
            | dict_type 
            | filelike_type
            | generator_type 
            | iterable_type
            | literal_type 
            | optional_type 
            | restricted_type 
            | set_type
            | tuple_type 
            | union_type 
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
        'COMPLEX': 'complex',
        'OBJECT': 'Any',
        'PATHLIKE': 'PathLike',
        'FILELIKE': 'FileLike'
    }

    def alt_type(self, tree):
        """
        alt_type: _LBRACE array_kind (_COMMA array_kind)* _RBRACE [_COMMA] ([shape_qualifier] | [shape]) [type_qualifier]
        """
        pass

    def array_type(self, tree):
        """
        array_type: [NDARRAY|NUMPY] basic_type [_DASH] array_kind [[_COMMA] (dimension | shape_qualifier)]
                | array_kind [_COMMA] shape_qualifier [[_COMMA] type_qualifier] 
                | array_kind [_COMMA] type_qualifier [[_COMMA] shape_qualifier] 
                | (dimension | shape) array_kind [type_qualifier]
                | array_kind
        """
        arr_type = ''
        elt_type = None
        imports = set()
        for child in tree.children:
            if isinstance(child, Token) and (child.type == 'NDARRAY' or child.type == 'NUMPY'):
                arr_type = 'NDArray'
                imports.add(('NDArray', 'numpy.typing'))
            if isinstance(child, Tree) and isinstance(child.data, Token):
                tok = child.data
                subrule = tok.value
                if subrule == 'array_kind':
                    arr_type, imp = self._visit_tree(child)
                    imports.update(imp)
                elif subrule == 'basic_type' or subrule == 'type_qualifier':
                    elt_type, imp = self._visit_tree(child)
                    imports.update(imp)
        if elt_type and arr_type != 'ArrayLike':
            arr_type += f'[{elt_type}]'
        return arr_type, imports

    def array_kind(self, tree):
        """
        array_kind: ARRAYLIKE 
          | LIST
          | NDARRAY 
          | MATRIX 
          | SEQUENCE
          | ARRAY 
          | ARRAYS 
        """
        arr_type = ''
        imports = set()
        for child in tree.children:
            if isinstance(child, Token):
                if child.type == 'LIST':
                    arr_type = 'list'
                elif child.type == 'SEQUENCE':
                    arr_type = 'Sequence'
                    imports.add(('Sequence', 'typing'))
                elif child.type == 'ARRAYLIKE':
                    arr_type = 'ArrayLike'
                    imports.add(('ArrayLike', 'numpy.typing'))
        if not arr_type:
            arr_type = 'NDArray'
            imports.add(('NDArray', 'numpy.typing'))

        return arr_type, imports

    def type_qualifier(self, tree):
        """
            type_qualifier: OF ARRAYS
              | OF type 
              | [OF] DTYPE [_EQUALS] (basic_type | QUALNAME) [TYPE]
              | _LBRACKET type _RBRACKET
              | _LPAREN type _RPAREN
        """
        imports = set()
        for child in tree.children:
            if isinstance(child, Tree):
                return self._visit_tree(child)
            elif isinstance(child, Token):
                if child.type == 'QUALNAME':
                    type = self.handle_qualname(child.value, imports)
                    return type, imports
        # OF ARRAYS falls through here
        imports.add(('ArrayLike', 'numpy.typing'))
        return 'ArrayLike', imports

    def basic_type(self, tree) -> tuple[str, set[str]|None]:
        """
        basic_type: ANY 
            | [POSITIVE|NEGATIVE] INT 
            | STR 
            | [POSITIVE|NEGATIVE] FLOAT [IN range] 
            | BOOL
            | SCALAR [VALUE]
            | COMPLEX
            | OBJECT
        """
        imports = set()
        for child in tree.children:
            if isinstance(child, Token):
                if child.type == 'ANY':
                    imports.add(('typing', 'Any'))
                    return 'Any', imports
                elif child.type == 'SCALAR':
                    imports.add((f'{self._tlmodule}._typing', 'Scalar'))
                    return 'Scalar', imports
                elif child.type == 'PATHLIKE':
                    imports.add(('os', 'PathLike'))
                elif child.type == 'FILELIKE':
                    imports.add(('typing', 'IO'))
                    return 'IO', imports
                if child.type in self._basic_types:
                    return self._basic_types[child.type], imports

        assert(False)

    def callable_type(self, tree):
        """ callable_type: CALLABLE [_LBRACKET _LBRACKET type_list _RBRACKET _COMMA type _RBRACKET] """
        # TODO: handle signature
        imports = set()
        imports.add(('Callable', 'typing'))
        return "Callable", imports

    def class_type(self, tree):
        """
        class_type: [CLASSMARKER] [_A|_AN] class_specifier [INSTANCE|OBJECT]
        """
        cname = ''
        for child in tree.children:
            if isinstance(child, Tree):
                return self._visit_tree(child)
        assert(False)
        
    def class_specifier(self, tree):
        """
        class_specifier: (INSTANCE|SUBCLASS) OF QUALNAME 
               | QUALNAME [_COMMA|_LPAREN] OR [_A] SUBCLASS [OF QUALNAME][_RPAREN]
               | QUALNAME [_COLON QUALNAME] (CLASS|SUBCLASS)
               | QUALNAME _DASH LIKE 
               | QUALNAME
        """
        imp = set()
        cname = ''
        for child in tree.children:
            if isinstance(child, Token) and child.type == 'QUALNAME':
                cname = child.value
        # Now we need to normalize the name and find the imports
        x = cname.rfind('.')
        if x > 0:
            imp.add((cname[x+1:], cname[:x]))
            cname = cname[x+1:]
        return cname, imp ## FOR NOW, BUT WE NEED TO FIGURE OUT THE IMPORT      

    def dict_type(self, tree):
        """
        dict_type: (MAPPING|DICT) (OF|FROM) (basic_type|QUALNAME) [TO (basic_type | QUALNAME | ANY)] 
         | (MAPPING|DICT) [_LBRACKET type _COMMA type _RBRACKET]
        """
        dict_type = ''
        from_type = None
        to_type = None
        imports = set()
        for child in tree.children:
            if isinstance(child, Token):
                if child.type == 'MAPPING':
                    dict_type = 'Mapping'
                    imports.add(('Mapping', 'typing'))
                elif child.type == 'DICT':
                    dict_type = 'dict'
                elif child.type == 'QUALNAME':
                    to_type = self.handle_qualname(child.value, imports)
                    if from_type is None:
                        from_type = to_type
            elif isinstance(child, Tree):
                to_type, imp = self._visit_tree(child)
                if imp:
                    imports.update(imp)
                if from_type is None:
                    from_type = to_type

        if from_type is not None:
            dict_type += f'[{from_type}, {to_type}]'

        return dict_type, imports

    def filelike_type(self, tree):
        """ filelike_type: [READABLE|WRITABLE] FILELIKE [TYPE] """
        imports = set()
        imports.add((f'{self._tlmodule}._typing', 'FileLike'))
        return 'FileLike', imports

    def generator_type(self, tree):
        """ generator_type: GENERATOR [OF type] """
        pass

    def iterable_type(self, tree):
        """ iterable_type: ITERABLE [OF type] """
        pass
    
    def literal_type(self, tree)-> tuple[str, set[str]|None]:
        """ literal_type: STRING | NUMBER | NONE | TRUE | FALSE """
        imports = set()
        imports.add(('Literal', 'typing'))
        assert(len(tree.children) == 1 and isinstance(tree.children[0], Token))
        tok = tree.children[0]
        type = tok.type
        if type == 'STRING' or type == 'NUMBER':
            return f'Literal:' + tok.value, imports
        if type == 'NONE':
            return 'None', set()
        if type == 'TRUE':
            return 'Literal:True', imports
        if type == 'FALSE':
            return 'Literal:False', imports
        assert(False)

    def optional_type(self, tree):
        """ optional_type: OPTIONAL [_LBRACKET type _RBRACKET]"""
        for child in tree.children:
            if isinstance(child, Tree):
                type, imports = self._visit_tree(child)
                type += '|None'
                return type, imports
        assert(False)

    def restricted_type(self, tree):
        """ 
        restricted_type: [ONE OF] _LBRACE (literal_type|STR) (_COMMA (literal_type|STR))* _RBRACE
        """
        imp = set()
        types = []
        values = []
        rtn = ''
        for child in tree.children:
            if isinstance(child, Tree):
                type, imports = self._visit_tree(child)
                imp.update(imports)
                if type.startswith('Literal:'):
                    values.append(type[8:])
                else:
                    types.append('None')
            elif isinstance(child, Token) and child.type == 'STR':
                types.append('str')
        if values:
            types.append(f'Literal[{",".join(values)}]')
        return '|'.join(types), imp

    def set_type(self, tree):
        """
        set_type: (FROZENSET|SET) _LBRACKET type _RBRACKET
         | [_A] (FROZENSET|SET) [OF type_list]
        """
        set_type = None
        elt_type = None
        imports = set()
        for child in tree.children:
            if isinstance(child, Token):
                if child.type == 'SET':
                    set_type = 'set'
                elif child.type == 'FROZENSET':
                    set_type = 'frozenset'
            elif isinstance(child, Tree):
                elt_type, imports = self._visit_tree(child)

        if elt_type:
            set_type += '[' + elt_type + ']'

        return set_type, imports

    def tuple_type(self, tree):
        """
        tuple_type: TUPLE 
          | TUPLE (OF|WITH) [NUMBER] type (OR type)*
          | [TUPLE] _LPAREN type (_COMMA type)* _RPAREN
          | [TUPLE] _LBRACKET type (_COMMA type)* _RBRACKET
          | [TUPLE] _LBRACE type (_COMMA type)* _RBRACE
        """
        types = []
        imp = set()
        repeating = False
        count = 1
        for child in tree.children:
            if isinstance(child, Tree):
                type, imports = self._visit_tree(child)
                types.append(type)
                imp.update(imports)
            elif isinstance(child, Token) and child.type in ['OF', 'WITH']:
                repeating = True
            elif isinstance(child, Token) and child.type == 'NUMBER':
                count = int(child.value)
        if types:
            if repeating:
                if count > 1:
                    types = ["|".join(types)] * count
                else:
                    return f'tuple[{"|".join(types)}, ...]', imp
            
            return f'tuple[{",".join(types)}]', imp
        return 'tuple', None

    def union_type(self, tree):
        """ union_type: UNION _LBRACKET type (_COMMA type)* _RBRACKET | type (AND type)+ """
        types = set()
        imports = set()
        for child in tree.children:
            if isinstance(child, Tree):
                type, imp = self._visit_tree(child)
                types.add(type)
                imports.update(imp)
        return '|'.join(types), imports
    
    
_lark = Lark(_grammar)
_norm = Normalizer('tlmod', 'mod')


def parse_type(s: str, modname: str|None = None) -> tuple[str, dict[str, list[str]]|None]:
    """ Parse a type description from a docstring, returning the normalized
        type and the set of required imports, or None if no imports are needed.
    """
    #try:
    if True:
        tree = _lark.parse(s)
        tree.pretty() # TODO: remove
        n = _norm.visit(tree)
        imps = None
        if n[1]:
            imps = {}
            for imp in n[1]:
                what, where = imp
                if where not in imps:
                    imps[where] = []
                imps[where].append(what)
            for where in imps.keys():
                imps[where] = sorted(imps[where])
        return n[0], imps
    #except Exception as e:
    #    print(e)
    #    return '', None
    

