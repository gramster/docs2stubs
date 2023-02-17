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


def _get_repr(typ, arraylike: bool = False, matrixlike: bool=False):
    if isinstance(typ, UnionType):
        return '|'.join([_get_repr(a) for a in typ.__args__])
    elif isinstance(typ, GenericAlias) and typ._name and typ.__args__:
        # List, Tuple, etc
        if arraylike and typ._name == 'List':
            return 'ArrayLike'
        return f'{_adjust_name(typ._name)}[{", ".join([_get_repr(a) for a in typ.__args__])}]'
    if arraylike and (typ == np.ndarray or typ == pd.Series):
        return 'ArrayLike'
    if matrixlike and typ in [np.ndarray, pd.DataFrame, scipy.sparse.spmatrix, scipy.sparse.csr_matrix, scipy.sparse.csc_matrix]: # type: ignore 
        return 'MatrixLike'
    if typ == np.int64 or typ == np.uint64:
        return 'Int'
    if typ == np.float32 or typ == np.float64:
        return 'Float'
    if typ == np.ndarray:
        return 'np.ndarray'  # As MonkeyType would render it ndarray
    typ = _type_repr(typ).replace('NoneType', 'None')
    # Remove module qualifications from classes
    typ = _qualname.sub('\\1', typ)
    return typ


def combine_types(sigtype: type, doctype: str|None) -> str:
    arraylike = doctype is not None and doctype.find('ArrayLike') >= 0
    matrixlike = doctype is not None and doctype.find('MatrixLike') >= 0
    # This relies heaviliy on typing module internals
    if isinstance(sigtype, UnionType):
        components = [_get_repr(a, arraylike, matrixlike) for a in sigtype.__args__] # type: ignore
    else:
        components = [_get_repr(sigtype, arraylike, matrixlike)]
    
    if doctype is not None:
        # Very simple parser to split apart the doctype union. If we find a '[' we
        # find and skip to closing ']', handling any nested '[' and ']' pairs.
        # Else we split on '|'.
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

    # Remove some redundant types
    if len(components) > 1:
        if 'Float' in components:
            components = [c for c in components if c not in ['Int', 'int', 'float']]
        elif 'Int' in components:
            components = [c for c in components if c != 'int']
        if 'str' in components and doctype and doctype.find('Literal') >= 0:
            components = [c for c in components if c !=  'str']
        if 'Any' in components:
            components = [c for c in components if c !=  'Any']
    components = [c for c in components if c != 'None']

    result = '|'.join(set(components))
    return result
