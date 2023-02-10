from lark.lark import Lark
from lark.tree import Tree
from lark.lexer import Token
from lark.visitors import Interpreter


# NOTE: imports must be tuples (what, where)! It's easy to get that wrong and invert them.

_grammar = r"""
start: type_list
type_list: [RETURNS] type ((_COMMA|OR|_COMMA OR) type)* [_PERIOD|_COMMA] [[_LPAREN] DEFAULT [_EQUALS|_COLON] literal_type [_RPAREN] [_PERIOD]]
type: array_type 
    | basic_type [TYPE]
    | callable_type 
    | class_type 
    | dict_type 
    | filelike_type
    | generator_type 
    | iterable_type
    | iterator_type
    | literal_type 
    | optional_type 
    | restricted_type 
    | set_type
    | tuple_type 
    | union_type 
    | _LESSTHAN type _GRTRTHAN
array_type: [NDARRAY|NUMPY] basic_type [_MINUS] array_kind [[_COMMA] (dimension | shape_qualifier)]
          | [dimension] array_kinds [_COMMA] shape_qualifier [[_COMMA] type_qualifier] 
          | [dimension] array_kinds [_COMMA] type_qualifier [[_COMMA] shape_qualifier] 
          | shape basic_type array_kind
          | dimension basic_type array_kind
          | shape_qualifier array_kind [type_qualifier]
          | type_qualifier array_kind [shape_qualifier]
          | [dimension] array_kind
array_kinds: array_kind | _LBRACE array_kind [ _COMMA array_kind]* _RBRACE
array_kind: [A|AN] [SPARSE | _LPAREN SPARSE _RPAREN] ARRAYLIKE
          | [A] LIST
          | [A|AN] NDARRAY 
          | [A] [SPARSE | _LPAREN SPARSE _RPAREN] MATRIX [CLASS]
          | [A] SPARSEMATRIX [CLASS]
          | [A] SEQUENCE
          | [A|AN] [SPARSE | _LPAREN SPARSE _RPAREN] ARRAY 
          | ARRAYS 
          | SPARSE
dimension: _DIM ((OR | _SLASH) _DIM)* 
        | (NUMBER|NAME) _X (NUMBER|NAME) 
        | _LPAREN (NUMBER|NAME) _COMMA [NUMBER|NAME] [_COMMA [NUMBER|NAME]] _RPAREN
        | ONED
        | TWOD
        | THREED
shape_qualifier: [[WITH|OF] SHAPE] [_EQUALS|OF] (SIZE|LENGTH) (QUALNAME|NUMBER|shape)
               | [[WITH|OF] SHAPE] [_EQUALS|OF] shape (OR shape)* [dimension]
               | SAME SHAPE AS QUALNAME
               | OF SHAPE QUALNAME
shape: (_LPAREN|_LBRACKET) shape_element (_COMMA shape_element)* _COMMA? (_RPAREN|_RBRACKET)
shape_element: (QUALNAME|NUMBER|_ELLIPSIS) [[_MINUS|_PLUS] NUMBER]
type_qualifier: OF (ARRAYS|ARRAYLIKE)
              | OF [NUMBER] type 
              | [OF] DTYPE [_EQUALS] (basic_type | QUALNAME) [TYPE]
              | _LBRACKET type _RBRACKET
              | _LPAREN type _RPAREN
basic_type.2: ANY 
            | [POSITIVE|NEGATIVE] INT [_GRTRTHAN NUMBER]
            | STR 
            | [POSITIVE|NEGATIVE] FLOAT [IN _LBRACKET NUMBER _COMMA NUMBER _RBRACKET] [_GRTRTHAN NUMBER]
            | BOOL
            | [NUMPY] SCALAR [VALUE]
            | COMPLEX [SCALAR]
            | OBJECT
            | FILELIKE
            | PATHLIKE
            | [NUMPY] DTYPE
callable_type: CALLABLE [_LBRACKET [_LBRACKET type_list _RBRACKET _COMMA] type _RBRACKET]
class_type: [CLASSMARKER] class_specifier [INSTANCE|OBJECT]
        | class_specifier [_COMMA|_LPAREN] OR SUBCLASS [_RPAREN]
        | class_specifier [_COMMA|_LPAREN] OR class_specifier[_RPAREN]
class_specifier: [A|AN] (INSTANCE|CLASS|SUBCLASS) OF QUALNAME 
        | QUALNAME (INSTANCE|CLASS|SUBCLASS)
        | QUALNAME [_COMMA|_LPAREN] OR [A|AN|ANOTHER] SUBCLASS [OF QUALNAME][_RPAREN]
        | QUALNAME [_COLON QUALNAME] [_MINUS LIKE]
dict_type: (MAPPING|DICT) (OF|FROM) (basic_type|qualname) [(TO|_ARROW) (basic_type|qualname)] 
         | (MAPPING|DICT) [_LBRACKET type _COMMA type _RBRACKET]
filelike_type: [READABLE|WRITABLE] FILELIKE [TYPE]
generator_type: GENERATOR [OF type]
iterable_type: ITERABLE [(OF|OVER) type]
         | ITERABLE _LPAREN type _RPAREN
iterator_type: ITERATOR [(OF|OVER) type]
         | ITERATOR _LPAREN type _RPAREN
literal_type: STRING | NUMBER | NONE | TRUE | FALSE
optional_type: OPTIONAL [_LBRACKET type _RBRACKET]
restricted_type: [(ONE OF)| STR] _LBRACE (literal_type|STR) ((_COMMA|OR) (literal_type|STR|_ELLIPSIS))* _RBRACE [INT|BOOL]
set_type: (FROZENSET|SET) _LBRACKET type _RBRACKET
         | (FROZENSET|SET) [OF type_list]
tuple_type: [shape] TUPLE [(OF|WITH) [NUMBER] type (OR type)*]
          | [TUPLE] _LPAREN type (_COMMA type)* _RPAREN [PAIRS]
          | [TUPLE] _LBRACKET type (_COMMA type)* _RBRACKET
union_type: UNION _LBRACKET type (_COMMA type)* _RBRACKET 
          | type (AND type)+
          | [TUPLE] _LBRACE type (_COMMA type)* _RBRACE
          | type (_PIPE type)*
qualname.0: QUALNAME


A.2:         "a"i
AN.2:        "an"i
AND.2:       "and"i
ANOTHER.2:   "another"i
ANY.2:       "any"i
ARRAYLIKE.2: "arraylike"i | "array-like"i | "array like"i | "array_like"i | "masked array"i
ARRAY.2:     "array"i
ARRAYS.2:    "arrays"i
AS.2:        "as"i
AXES.2:      "axes"i
BOOL.2:      "bool"i | "bools"i | "boolean"i | "booleans"i
CALLABLE.2:  "callable"i | "callables"i | "function"i
CLASS.2:     "class"i
CLASSMARKER.2:":class:"
COLOR.2:     "color"i | "colors"i
COMPLEX.2:   "complex"i
DEFAULT.2:   "default"i
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
INT.2:       "int"| "ints"|  "integer" | "integers" | "int8" | "int16" | "int32" | "int64" | "uint8"| "uint16" | "uint32" | "uint64"
ITERABLE.2:  "iterable"i
ITERATOR.2:  "iterator"i | "iter"i
LENGTH.2:    "length"i
LIKE.2:      "like"i
LIST.2:      "list"i | "list thereof"i
MAPPING.2:   "mapping"i
MATPLOTLIB:  "matplotlib"i
MATRIX.2:    "matrix"i
NDARRAY.2:   "ndarray"i | "ndarrays"i | "nd-array"i | "numpy array"i | "np.array"i | "numpy.ndarray"i
NEGATIVE.2:  "negative"i
NONE.2:      "none"i
NUMPY.2:     "numpy"i
OBJECT.2:    "object"i | "objects"i
OF.2:        "of"i
ONE.2:       "one"i
ONED.2:      "1-d"i | "1d"i | "one-dimensional"i
OPTIONAL.2:  "optional"i
OR.2:        "or"i
OVER.2:      "over"i
PAIRS.2:     "pairs"i
PATHLIKE.2:  "path-like"i | "pathlike"i
POSITIVE.2:  "positive"i | "non-negative"i | "nonnegative"i
PRIVATE.2:   "private"i
READABLE.2:  "readable"i | "readonly"i | "read-only"i
RETURNS.2:   "returns"i
SAME.2:      "same"i
SCALAR.2:     "scalar"i
SEQUENCE.2:  "sequence"i | "sequence thereof"i
SET.2:       "set"i
SHAPE.2:     "shape"i
SIZE.2:      "size"i
SPARSE.2:    "sparse"i
SPARSEMATRIX.2: "sparse-matrix"i
STR.2:       "str"i | "string"i | "strings"i | "python string"i | "python str"i
SUBCLASS.2:  "subclass"i | "subclass thereof"i
THREED.2:    "3-d"i | "3d"i | "three-dimensional"i
TO.2:        "to"i
TRUE.2:      "true"i
TUPLE.2:     "tuple"i | "2-tuple"i | "2 tuple"i | "3-tuple"i | "3 tuple"i | "4-tuple" | "4 tuple" | "tuple thereof"i
TWOD.2:      "2-d"i | "2d"i | "two-dimensional"i
TYPE.2:      "type"i
UNION.2:     "union"i
VALUE.2:     "value"i
WITH.2:      "with"i
WRITABLE.2:  "writeable"i | "writable"i


_ARROW:     "->"
_ASTERISK:  "*"
_BACKTICK:  "`"
_C_CONTIGUOUS: "C-contiguous"i
_COLON:    ":"
_COMMA:    ","
_DIM:      "0-d"i | "1-d"i | "2-d"i | "3-d"i | "1d"i | "2d"i | "3d"i
_ELLIPSIS: "..."
_EQUALS:   "="
_GRTRTHAN:  ">"
_LBRACE:   "{"
_LBRACKET:  "["
_LESSTHAN:  "<"
_LPAREN:    "("
_MINUS:     "-"
_NEWLINE:   "\n"
_PIPE:      "|"
_PLURAL:    "\\s"
_PLUS:      "+"
_PERIOD:   "."
_PRIVATE:  "private"
_RBRACE:   "}"
_RBRACKET:  "]"
_RPAREN:    ")"
_SLASH:     "/"
_STRIDED:   "strided"i
_SUCH:      "such"
_THE:       "the"
_TILDE:     "~"
_X:         "x"


NAME:      /[A-Za-z_][A-Za-z0-9_\-]*/
NUMBER:    /-?[0-9][0-9\.]*e?\-?[0-9]*/
QNAME:  /\.?[A-Za-z_][A-Za-z_0-9\-]*(\.[A-Za-z_.][A-Za-z0-9_\-]*)*/
QUALNAME:  QNAME | MATPLOTLIB AXES | MATPLOTLIB COLOR
STRINGSQ:  /\'[^\']*\'/
STRINGDQ:  /\"[^\"]*\"/
STRING:    STRINGSQ | STRINGDQ

%import common.WS
%ignore WS

%ignore _ASTERISK
%ignore _BACKTICK
%ignore _C_CONTIGUOUS
%ignore _PLURAL
%ignore _PRIVATE
%ignore _STRIDED
%ignore _SUCH
%ignore _THE
%ignore _TILDE
"""

class Normalizer(Interpreter):
    def configure(self, module:str|None, classes: dict|None, is_param:bool):
        if module is None:
            module = ''
        x = module.find('.')
        if x >= 0:
            self._tlmodule = module[:x]  # top-level module
        else:
            self._tlmodule = module
        self._module = module
        self._classes = classes
        self._is_param = is_param

    def handle_qualname(self, name: str, imports: set) -> str:
        return name
    
    def start(self, tree) -> tuple[str, set[str]]:
        """ start: type_list [PERIOD] """
        result = self.visit(tree.children[0])
        return result
        
    def type_list(self, tree) -> tuple[str, set[str]]:
        """ type_list: type ((_COMMA|OR|_COMMA OR) type)*  [OR NONE] [_PERIOD] """
        types = [] # We want to preserve order so don't use a set
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
                        type = result[0]
                        if type not in types:
                            types.append(type)
                    if result[1]:
                        imports.update(result[1])

        if not imports:
            imports = set()
        if literals:
            type = 'Literal[' + ','.join(literals) + ']'
            if type not in types:
                types.append(type)
        if has_none:
            type = 'None'
            if type not in types:
                types.append(type)
        type = '|'.join(types)
        return type, imports

    def type(self, tree)-> tuple[str, set[str]]:
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
            #print(child)
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
        'FILELIKE': 'IO'
    }
    _basic_types_imports = {
        'ANY': 'typing',
        'SCALAR': '_typing',
        'OBJECT': 'typing',
        'PATHLIKE': 'os',
        'FILELIKE': 'typing'
    }

    def array_type(self, tree) -> tuple[str, set[tuple[str, str]]]:
        arr_types = set()
        elt_type = None
        imports = set()
        for child in tree.children:
            if isinstance(child, Token) and (child.type == 'NDARRAY' or child.type == 'NUMPY'):
                if self._is_param:
                    arr_type = 'ArrayLike'
                    imports.add(('ArrayLike', 'numpy.typing'))
                else:
                    # TODO: if we have an elt type, we can use NDArray from numpy.typing
                    arr_types.add('np.ndarray')
                    imports.add(('', 'numpy'))
            elif isinstance(child, Tree) and isinstance(child.data, Token):
                tok = child.data
                subrule = tok.value
                if subrule == 'array_kinds':
                    types, imp = self._visit_tree(child)
                    arr_types.update(types)
                    imports.update(imp)
                elif subrule == 'array_kind':
                    type, imp = self._visit_tree(child)
                    arr_types.add(type)
                    imports.update(imp)
                elif subrule == 'basic_type' or subrule == 'type_qualifier':
                    elt_type, imp = self._visit_tree(child)
                    imports.update(imp)
        if elt_type:
            if self._is_param and 'list' in arr_types:
                arr_types.add('Sequence')
                arr_types.remove('list')
                imports.add(('Sequence', 'typing'))
            return '|'.join([f'{typ}[{elt_type}]' if typ in ['Sequence', 'list', 'ndarray'] else f'{typ}' \
                    for typ in arr_types]), imports
        else:
            return '|'.join(arr_types), imports

    def array_kinds(self, tree) -> tuple[set[str], set[tuple[str, str]]]:
        imports = set()
        types = set()
        for child in tree.children:
            if isinstance(child, Tree):
                type, imp = self._visit_tree(child)
                imports.update(imp)
                types.add(type)
        return types, imports

    def array_kind(self, tree) -> tuple[str, set[tuple[str, str]]]:
        arr_type = ''
        imports = set()
        is_sparse = False
        for child in tree.children:
            if isinstance(child, Token):
                if child.type == 'SPARSE':
                    is_sparse = True
                    continue
                elif child.type == 'NDARRAY':
                    # TODO: if we have an elt type, we can use NDArray from numpy.typing
                    if self._is_param:
                        arr_type = 'ArrayLike'
                        imports.add(('ArrayLike', 'numpy.typing'))
                    else:
                        arr_type = 'np.ndarray'
                        imports.add(('', 'numpy'))
                elif child.type == 'SEQUENCE':
                    arr_type = 'Sequence'
                    imports.add(('Sequence', 'typing'))
                elif self._is_param or child.type == 'ARRAYLIKE':
                    arr_type = 'ArrayLike'
                    imports.add(('ArrayLike', 'numpy.typing'))
                elif child.type == 'LIST':
                    arr_type = 'list'
                elif child.type == 'SPARSEMATRIX' or child.type == 'MATRIX':
                    if self._is_param:
                        arr_type = 'MatrixLike'
                        imports.add(('MatrixLike', f'{self._tlmodule}._typing'))
                    else:
                        arr_type = 'spmatrix'
                        imports.add(('spmatrix', 'scipy.sparse'))
                break

        if not arr_type:
            if self._is_param:
                arr_type = 'ArrayLike'
                imports.add(('ArrayLike', 'numpy.typing'))
            else:
                arr_type = 'np.ndarray'
                imports.add(('', 'numpy'))

        return arr_type, imports

    def type_qualifier(self, tree) -> tuple[str, set[tuple[str, str]]]:
        """
        type_qualifier: OF (ARRAYS|ARRAYLIKE)
              | OF [NUMBER] type 
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

    def basic_type(self, tree) -> tuple[str, set[str]]:
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
                    imports.add(('Any', 'typing'))
                    return 'Any', imports
                elif child.type == 'SCALAR':
                    imports.add(('Scalar', f'{self._tlmodule}._typing'))
                    return 'Scalar', imports
                elif child.type == 'PATHLIKE':
                    imports.add(('PathLike', 'os'))
                elif child.type == 'FILELIKE':
                    imports.add(('IO', 'typing'))
                    return 'IO', imports
                if child.type in self._basic_types:
                    typ = self._basic_types[child.type]
                    if child.type in self._basic_types_imports:
                        imp = self._basic_types_imports[child.type]
                        if imp == '_typing':
                            imp = self._tlmodule + '.' + imp
                        imports.add((typ, imp))
                    return typ, imports

        assert(False)

    def callable_type(self, tree) -> tuple[str, set[tuple[str, str]]]:
        """ callable_type: CALLABLE [_LBRACKET _LBRACKET type_list _RBRACKET _COMMA type _RBRACKET] """
        # TODO: handle signature
        imports = set()
        imports.add(('Callable', 'typing'))
        return "Callable", imports

    def class_type(self, tree) -> tuple[str, set[tuple[str, str]]]:
        """
        class_type: [CLASSMARKER] [_A|_AN] class_specifier [INSTANCE|OBJECT]
        """
        cname = ''
        for child in tree.children:
            if isinstance(child, Tree):
                return self._visit_tree(child)
        assert(False)
        
    def class_specifier(self, tree) -> tuple[str, set[tuple[str, str]]]:
        """
        class_specifier: (INSTANCE|SUBCLASS) OF QUALNAME 
               | QUALNAME [_COMMA|_LPAREN] OR [_A] SUBCLASS [OF QUALNAME][_RPAREN]
               | QUALNAME [_COLON QUALNAME] (CLASS|SUBCLASS)
               | QUALNAME _MINUS LIKE 
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
        elif self._classes and cname in self._classes:
            imp.add((cname, self._classes[cname]))

        return cname, imp      

    def dict_type(self, tree) -> tuple[str, set[tuple[str, str]]]:
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
                    if self._is_param:
                        dict_type = 'Mapping'
                        imports.add(('Mapping', 'typing'))
                    else:
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

    
    def qualname(self, tree) -> tuple[str, set[tuple[str, str]]]:
        """
        qualname: QUALNAME
        This is mostly to give qualnames a lower priority than other types like basic_type
        """
        for child in tree.children:
            if isinstance(child, Token):
                imports = set()
                return self.handle_qualname(child.value, imports), imports
        raise Exception("qualname didn't find a token")
            

    def filelike_type(self, tree) -> tuple[str, set[tuple[str, str]]]:
        """ filelike_type: [READABLE|WRITABLE] FILELIKE [TYPE] """
        imports = set()
        imports.add(('FileLike', f'{self._tlmodule}._typing'))
        return 'FileLike', imports

    def generator_type(self, tree) -> tuple[str, set[tuple[str, str]]]:
        """ generator_type: GENERATOR [OF type] """
        # TODO: the type
        imports = set()
        imports.add(('Generator', 'collections.abc'))
        for child in tree.children:
            if isinstance(child, Tree):
                type, imp = self._visit_tree(child)
                if type:
                    imports.update(imp)
                    return f'Generator[{type}, None, None]', imports
        return 'Generator', imports

    def iterable_type(self, tree) -> tuple[str, set[tuple[str, str]]]:
        """ iterable_type: ITERABLE [OF type] """
        imports = set()
        imports.add(('Iterable', 'collections.abc'))
        for child in tree.children:
            if isinstance(child, Tree):
                type, imp = self._visit_tree(child)
                if type:
                    imports.update(imp)
                    return f'Iterable[{type}]', imports
        return 'Iterable', imports
    
    def iterator_type(self, tree) -> tuple[str, set[tuple[str, str]]]:
        """ iterator_type: ITERATOR [OVER type] """
        imports = set()
        imports.add(('Iterator', 'collections.abc'))
        for child in tree.children:
            if isinstance(child, Tree):
                type, imp = self._visit_tree(child)
                if type:
                    imports.update(imp)
                    return f'Iterator[{type}]', imports
        return 'Iterator', imports
    
    def literal_type(self, tree)-> tuple[str, set[str]]:
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

    def optional_type(self, tree) -> tuple[str, set[tuple[str, str]]]:
        """ optional_type: OPTIONAL [_LBRACKET type _RBRACKET]"""
        for child in tree.children:
            if isinstance(child, Tree):
                type, imports = self._visit_tree(child)
                type += '|None'
                return type, imports
        return 'None', set()

    def restricted_type(self, tree) -> tuple[str, set[tuple[str, str]]]:
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
            imp.add(('Literal', 'typing'))
        return '|'.join(types), imp

    def set_type(self, tree) -> tuple[str, set[tuple[str, str]]]:
        """
        set_type: (FROZENSET|SET) _LBRACKET type _RBRACKET
         | [_A] (FROZENSET|SET) [OF type_list]
        """
        set_type = 'set'
        elt_type = None
        imports = set()
        for child in tree.children:
            if isinstance(child, Token):
                if child.type == 'FROZENSET':
                    set_type = 'frozenset'
            elif isinstance(child, Tree):
                elt_type, imports = self._visit_tree(child)

        if elt_type:
            set_type += '[' + elt_type + ']'

        return set_type, imports

    def tuple_type(self, tree) -> tuple[str, set[tuple[str, str]]]:
        """
        tuple_type: [shape] TUPLE (OF|WITH) [NUMBER] type (OR type)*
          | [TUPLE] _LPAREN type (_COMMA type)* _RPAREN
          | [TUPLE] _LBRACKET type (_COMMA type)* _RBRACKET
          | [TUPLE] _LBRACE type (_COMMA type)* _RBRACE
        """
        types = []
        imp = set()
        repeating = False
        count = 1
        has_shape = False
        for child in tree.children:
            if isinstance(child, Tree):
                type, imports = self._visit_tree(child)
                if type is None:
                    has_shape = True
                else:
                    types.append(type)
                    imp.update(imports)
            elif isinstance(child, Token) and child.type in ['OF', 'WITH']:
                repeating = True
            elif isinstance(child, Token) and child.type == 'NUMBER':
                count = int(child.value)
        if has_shape:
            return 'tuple', set()
        if types:
            if repeating:
                if count > 1:
                    types = ["|".join(types)] * count
                else:
                    return f'tuple[{"|".join(types)}, ...]', imp
            
            return f'tuple[{",".join(types)}]', imp
        return 'tuple', set()

    def union_type(self, tree) -> tuple[str, set[tuple[str, str]]]:
        """ union_type: UNION _LBRACKET type (_COMMA type)* _RBRACKET | type (AND type)+ """
        types = set()
        imports = set()
        for child in tree.children:
            if isinstance(child, Tree):
                type, imp = self._visit_tree(child)
                types.add(type)
                imports.update(imp)
        return '|'.join(types), imports
    
    def shape(self, tree):
        return None, None
    

_lark = Lark(_grammar)
_norm =  _norm = Normalizer()

    
def parse_type(s: str, modname: str|None = None, classes: dict|None = None, is_param:bool=False) -> tuple[str, dict[str, list[str]]]:
    """ Parse a type description from a docstring, returning the normalized
        type and the set of required imports, or None if no imports are needed.
    """
    try:
    #if True:
        tree = _lark.parse(s)
        #print(tree.pretty()) # TODO: remove
        _norm.configure(modname, classes, is_param)
        n = _norm.visit(tree)
        imps = {}
        if n[1]:
            for imp in n[1]:
                what, where = imp
                if where not in imps:
                    imps[where] = []
                if what:
                    imps[where].append(what)
            for where in imps.keys():
                imps[where] = sorted(imps[where])
        return n[0], imps
    except Exception as e:
        #print(e)
        return s, {}
    

