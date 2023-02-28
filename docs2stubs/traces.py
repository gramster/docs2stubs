# Utilities for working with MonkeyType traces.
# Borrows heavily from the MonkeyType code. However, as
# I want to incorporate this into the docs2stubs CST tree
# walker, I want to be able to access the traced types directly.


from typing import (
    IO, Optional, Tuple, List, Dict, Any, Iterable, cast, Type, 
    _GenericAlias as GenericAlias, _UnionGenericAlias as UnionType, _type_repr # type: ignore
)


# A variant of the SQL store that uses counts instead of timestamps.
# This is duplicated in the tracing directory, so need to refactor
# at some point to avoid that.

import datetime
from inspect import Signature
import os
import re
import sqlite3
import sys
import atexit
from typing import Iterable, List, Optional, Tuple, Union

from monkeytype.db.base import CallTraceStore, CallTraceThunk
from monkeytype.encoding import CallTraceRow, serialize_traces
from monkeytype.tracing import CallTrace
from monkeytype.config import DefaultConfig
from monkeytype.stubs import build_module_stubs_from_traces
from monkeytype.typing import shrink_types

import numpy as np
import pandas as pd
import scipy


DEFAULT_TABLE = "monkeytype_call_traces"
QueryValue = Union[str, int]
ParameterizedQuery = Tuple[str, List[QueryValue]]


def create_call_trace_table(
    conn: sqlite3.Connection, table: str = DEFAULT_TABLE
) -> None:
    query = """
CREATE TABLE IF NOT EXISTS {table} (
  count       INTEGER,
  module      TEXT,
  qualname    TEXT,
  arg_types   TEXT,
  return_type TEXT,
  yield_type  TEXT);
""".format(
        table=table
    )
    with conn:
        conn.execute(query)


def make_query(
    table: str, module: str, qualname: Optional[str], limit: int
) -> ParameterizedQuery:
    raw_query = """
    SELECT
        module, qualname, arg_types, return_type, yield_type
    FROM {table}
    WHERE
        module == ?
    """.format(
        table=table
    )
    values: List[QueryValue] = [module]
    if qualname is not None:
        raw_query += " AND qualname LIKE ? || '%'"
        values.append(qualname)
    raw_query += """
    GROUP BY
        module, qualname, arg_types, return_type, yield_type
    ORDER BY count DESC
    LIMIT ?
    """
    values.append(limit)
    return raw_query, values



class SQLiteDedupStore(CallTraceStore):

    def __init__(self, conn: sqlite3.Connection, table: str = DEFAULT_TABLE) -> None:
        self.conn = conn
        self.table = table
        self.count = 0

    @classmethod
    def make_store(cls, connection_string: str) -> "CallTraceStore":
        conn = sqlite3.connect(connection_string)
        create_call_trace_table(conn)
        return cls(conn)

    def add(self, traces: Iterable[CallTrace]) -> None:
        values = []
        for row in serialize_traces(traces):
            values.append(
                (
                    row.module,
                    row.qualname,
                    row.arg_types,
                    row.return_type,
                    row.yield_type,
                )
            )
        if len(values):
            self.count += len(values)
            with self.conn:
                self.conn.executemany(f"INSERT INTO {self.table} VALUES (1, ?, ?, ?, ?, ?)", values)


    def filter(
        self, module: str, qualname_prefix: Optional[str] = None, limit: int = 2000
    ) -> List[CallTraceThunk]:
        sql_query, values = make_query(self.table, module, qualname_prefix, limit)
        with self.conn:
            cur = self.conn.cursor()
            cur.execute(sql_query, values)
            return [CallTraceRow(*row) for row in cur.fetchall()]

    def list_modules(self) -> List[str]:
        with self.conn:
            cur = self.conn.cursor()
            cur.execute(
                """
                        SELECT module FROM {table}
                        GROUP BY module
                        """.format(
                    table=self.table
                )
            )
            return [row[0] for row in cur.fetchall() if row[0]]


def _load_module_traces(
    db, 
    module, stderr: IO[str], verbose: bool = False
) -> list[CallTrace]|None:
    thunks = SQLiteDedupStore.make_store(db).filter(module)
    traces = []
    failed_to_decode_count = 0
    for thunk in thunks:
        try:
            traces.append(thunk.to_trace())
        except Exception as e:
            if verbose:
                print(f"WARNING: Failed decoding trace: {e}", file=stderr)
            failed_to_decode_count += 1
    if failed_to_decode_count and not verbose:
        print(
            f"{failed_to_decode_count} traces failed to decode; use -v for details",
            file=stderr,
        )
    return traces


MAX_TYPED_DICT_SIZE = 128
_stubs = {}
_dbpath = ''
_pkg = ''


def init_trace_loader(dbpath: str, pkg: str):
    global _dbpath, _pkg
    _dbpath = dbpath
    _pkg = pkg


def _get_module_stubs(dbpath: str, pkg: str, module: str):
    if _dbpath == '':
        raise Exception("get_module_traces called before init_trace_loader")
    if module not in _stubs:
        _stubs[module] = None
        traces = _load_module_traces(f'{dbpath}/{pkg}.sqlite3', module, sys.stderr)
        if traces:
            stubs = build_module_stubs_from_traces(traces, MAX_TYPED_DICT_SIZE)
            if stubs:
                _stubs[module] = stubs[module]
    return _stubs[module]


def get_toplevel_function_signature(module: str, function: str) -> Signature|None:
    stubs = _get_module_stubs(_dbpath, _pkg, module)
    if stubs:
        try:
            return stubs.function_stubs[function].signature
        except:
            return None
    return None

def get_method_signature(module: str, class_: str, method: str) -> Signature|None:
    stubs = _get_module_stubs(_dbpath, _pkg, module)
    if stubs:
        try:
            return stubs.class_stubs[class_].function_stubs[method].signature
        except:
            return None
    return None

def simplify_types(ts: set[Type]):
    return shrink_types(ts, MAX_TYPED_DICT_SIZE)

def render_type(t: Type) -> str:
    return '';


_qualname = re.compile(r'[A-Za-z_\.]*\.([A-Za-z_][A-Za-z_0-9]*)')


def _adjust_name(name: str) -> str:
    if name in ['List', 'Dict', 'Tuple', 'Set']:
        return name.lower()
    return name


def _get_repr(tlmodule: str, typ, arraylike: bool = False, matrixlike: bool=False) -> tuple[str, set[tuple[str, str]]]:
    imports = set()
    if isinstance(typ, UnionType):
        components = []
        for a in typ.__args__:
            t, i = _get_repr(tlmodule, a)
            components.append(t)
            imports.update(i)
        typ = '|'.join(components)
    elif isinstance(typ, GenericAlias) and typ._name and typ.__args__:
        # List, Tuple, etc
        if arraylike and typ._name == 'List':
            imports.add(('ArrayLike', f'{tlmodule}._typing'))
            typ = 'ArrayLike'
        else:
            components = []
            for a in typ.__args__:
                t, i = _get_repr(tlmodule, a)
                components.append(t)
                imports.update(i)
            typ = f'{_adjust_name(typ._name)}[{", ".join(components)}]'
    elif arraylike and (typ == np.ndarray or typ == pd.Series):
        imports.add(('ArrayLike', f'{tlmodule}._typing'))
        typ = 'ArrayLike'
    elif matrixlike and typ in [np.ndarray, pd.DataFrame, scipy.sparse.spmatrix, scipy.sparse.csr_matrix, scipy.sparse.csc_matrix]: # type: ignore 
        imports.add(('MatrixLike', f'{tlmodule}._typing'))
        typ = 'MatrixLike'
    elif typ == np.int64 or typ == np.uint64:
        imports.add(('Int', f'{tlmodule}._typing'))
        typ =  'Int'
    elif typ == np.float32 or typ == np.float64:
        imports.add(('Float', f'{tlmodule}._typing'))
        typ = 'Float'
    else:
        typ = _type_repr(typ).replace('NoneType', 'None')
        if typ.find('[') < 0 and typ.find('.') > 0:
            module, name = typ.rsplit('.', 1)
            imports.add((name, module))
            typ = name

    return typ, imports

    
def combine_types(tlmodule: str, sigtype: type|None, doctype: str|None, valtyp: str|None) -> tuple[str, set[tuple[str, str]]]:

    arraylike = doctype is not None and doctype.find('ArrayLike') >= 0
    matrixlike = doctype is not None and doctype.find('MatrixLike') >= 0
    imports = set()
    components = []

    # This relies on typing module internals
    if isinstance(sigtype, UnionType):
        for a in sigtype.__args__:  # type: ignore
            t, i = _get_repr(tlmodule, a, arraylike, matrixlike)
            components.append(t)
            imports.update(i)
        if len(components) > 5:
            # TODO: if components is very long, see if there are classes that have a common
            # base class and use that instead. Ideally go down structurally into components
            # (e.g. dict[str, A] and dict[str, B] could be dict[str, A|B] and then dict[str, base(A, B)]).
            # For now we just make really long annotations into Any.
            pass
    elif sigtype is not None:
        t, i = _get_repr(tlmodule, sigtype, arraylike, matrixlike)
        components = [t]
        imports.update(i)
    
    if doctype is not None:
        # Very simple parser to split apart the doctype union. If we find a '[' we
        # find and skip to closing ']', handling any nested '[' and ']' pairs.
        # Else we split on '|'.
        # Note: in theory we should have already added the imports for these so 
        # don't need to do it here.
        i = 0
        start = 0
        while i < len(doctype):
            if doctype[i] == '[':
                i += 1
                depth = 1
                while i < len(doctype) and depth > 0:
                    if doctype[i] == '[':
                        depth += 1
                    elif doctype[i] == ']':
                        depth -= 1
                    i += 1
            else:
                if doctype[i] == '|':
                    components.append(doctype[start:i])
                    start = i + 1
                i += 1
        components.append(doctype[start:i])

    if 'Any' in components:
        components = ['Any']
        imports.add(('Any', 'typing'))
        return 'Any', imports
    
    # Include the default value type
    if valtyp is not None:
        components.append(valtyp)

    if len(components) > 1:
        # Remove some redundant types
        if 'Float' in components:
            components = [c for c in components if c not in ['Int', 'int', 'float']]
            imports.add(('Float', f'{tlmodule}._typing'))
        elif 'Int' in components:
            components = [c for c in components if c != 'int']
            imports.add(('Int', f'{tlmodule}._typing'))


        if 'str' in components and doctype and doctype.find('Literal') >= 0:
            # Remove str and fold in the literals into one
            newc = []
            lits = []
            for c in components:
                if c.startswith('Literal['):
                    lits.append(c[8:-1].replace('"', "'"))
                elif c != 'str':
                    newc.append(c)
            newvals = ', '.join(set(lits))
            if len(newvals.split(',')) < 2: # Less than two options means we're probably missing details; fall back to str
                newc.append('str')
            else:
                newc.append(f'Literal[{newvals}]')
            components = newc

    # Replace Literal with str if there is only one option. Else 
    # remove str if there are more than one option.
    newc = []
    lits = []
    has_str = False
    has_literal = False
    for c in components:
        if c.find('[') >= 0:
            if c.startswith('Literal['):
                has_literal = True
                lits.append(c[8:-1])
            elif c == 'str':
                has_str = True
            else:
                newc.append(c)
        else:
            newc.append(c)
            if c.find('.') > 0:
                module, name = c.rsplit('.', 1)
                if module == 'np':
                    module = 'numpy'
                elif module == 'pd':
                    module = 'pandas'
                imports.add((name, module))

    newlitvals = ', '.join(lits)
    if len(newlitvals.split(',')) >= 2:
        newc.append(f'Literal[{newlitvals}]')
    elif has_str or has_literal:
       newc.append('str')
    components = newc

    result = '|'.join(set(components))

    if False and len(result) >= 200:  # remove False to enable this 
        # Somewhat arbitrary cutoff to avoid some pathological cases
        result = 'Any'
        imports = set()

    else:
        # Kludge: fix some possibly missing imports. Shouldn't be needed if everything
        # else worked :-(
        # TODO: use a regexp to extract identifiers instead of looping through each time.
        # TODO: we could do _all_ the import collecting here and leave it out elsewhere.

        extras = {
            'RandomState': 'numpy.random',
            'ArrayLike': f'{tlmodule}._typing',
            'MatrixLike': f'{tlmodule}._typing',
            'Float': f'{tlmodule}._typing',
            'Int': f'{tlmodule}._typing',
            'BaseEstimator': 'sklearn.base',
            'Classifier': f'{tlmodule}._typing',
            'Estimator': f'{tlmodule}._typing',
            'Regressor': f'{tlmodule}._typing',
            'Axes': 'matplotlib.axes',
            'Figure': 'matplotlib.figure',
            'Memory': 'joblib',
            'DType': 'numpy',
            'MinCovDet': 'sklearn.covariance',
            'Colormap': 'matplotlib.colors',

        }

        for k, v in extras.items():
            if result.find(k) >= 0:
                imports.add((k, v))

        # TODO: check the following types: Label, Batch, module, sklearn.cluster.
        # They are being used without the needed imports.
        # from ._random import sample_without_replacement
        # class loguniform(scipy.stats.reciprocal):
        # from .murmurhash import murmurhash3_32 as murmurhash3_32



    return result, imports
