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

from inspect import Signature
import re
import sqlite3
import sys
from typing import Iterable

from monkeytype.db.base import CallTraceStore, CallTraceThunk
from monkeytype.encoding import CallTraceRow, serialize_traces
from monkeytype.tracing import CallTrace
from monkeytype.stubs import build_module_stubs_from_traces
from monkeytype.typing import shrink_types

import numpy as np
import pandas as pd
import scipy


DEFAULT_TABLE = "monkeytype_call_traces"
QueryValue = str|int
ParameterizedQuery = tuple[str, list[QueryValue]]


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



_qualname = re.compile(r'[A-Za-z_\.]*\.([A-Za-z_][A-Za-z_0-9]*)')


def _adjust_name(name: str) -> str:
    if name in ['List', 'Dict', 'Tuple', 'Set']:
        return name.lower()
    return name


def _get_repr(tlmodule: str, typ, arraylike: bool = False, matrixlike: bool=False) -> str:
    if isinstance(typ, UnionType):
        components = []
        for a in typ.__args__:
            t  = _get_repr(tlmodule, a)
            components.append(t)
        typ = '|'.join(components)
    elif isinstance(typ, GenericAlias) and typ._name and typ.__args__:
        # List, Tuple, etc
        if arraylike and typ._name == 'List':
            typ = 'ArrayLike'
        else:
            components = []
            for a in typ.__args__:
                t = _get_repr(tlmodule, a)
                components.append(t)
            typ = f'{_adjust_name(typ._name)}[{", ".join(components)}]'
    elif arraylike and (typ == np.ndarray or typ == pd.Series):
        typ = 'ArrayLike'
    elif matrixlike and typ in [np.ndarray, pd.DataFrame, scipy.sparse.spmatrix, scipy.sparse.csr_matrix, scipy.sparse.csc_matrix]: # type: ignore 
        typ = 'MatrixLike'
    elif typ == np.int64 or typ == np.uint64:
        typ =  'Int'
    elif typ == np.float32 or typ == np.float64:
        typ = 'Float'
    else:
        typ = _type_repr(typ).replace('NoneType', 'None')

    return typ

    
qualname_re = re.compile(r'[A-Za-z_][A-Za-z0-9_\.]*[A-Za-z0-9_]*')
    

def combine_types(tlmodule: str, sigtype: type|None, doctype: str|None, valtyp: str|None) -> tuple[str, set[tuple[str, str]]]:

    arraylike = doctype is not None and doctype.find('ArrayLike') >= 0
    matrixlike = doctype is not None and doctype.find('MatrixLike') >= 0
    imports = set()
    components = []

    # This relies on typing module internals
    if isinstance(sigtype, UnionType):
        for a in sigtype.__args__:  # type: ignore
            t = _get_repr(tlmodule, a, arraylike, matrixlike)
            components.append(t)
        if len(components) > 5:
            # TODO: if components is very long, see if there are classes that have a common
            # base class and use that instead. Ideally go down structurally into components
            # (e.g. dict[str, A] and dict[str, B] could be dict[str, A|B] and then dict[str, base(A, B)]).
            # For now we just make really long annotations into Any.
            pass
    elif sigtype is not None:
        t = _get_repr(tlmodule, sigtype, arraylike, matrixlike)
        components = [t]
    
    # Check how long this type is and if it's too long, just drop the trace part, as that
    # is typically the cause, plus we assume the docstring mappings are desired and not
    # something we should be discarding, while the trace aujgmentation is 'gravy'.
    # The threshhold here is a bit arbitrary but assume we want the type to fit comfortably
    # on one line.
    if sum(len(c) for c in components) > 65:
        components = []
        imports = set()

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
        elif 'Int' in components:
            components = [c for c in components if c != 'int']

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

    newlitvals = ', '.join(lits)
    if len(newlitvals.split(',')) >= 2:
        newc.append(f'Literal[{newlitvals}]')
    elif has_str or has_literal:
       newc.append('str')

    components = set()
    for c in set(newc):
        # Now go through all the qualnames in the types, and replace with last part,
        # adding an import to the needed imports.
        typing = f'{tlmodule}._typing'
        for x in qualname_re.findall(c):
            if x.find('.') >= 0:
                parts = x.split('.')
                mod = '.'.join(parts[:-1])
                name = parts[-1]
                imports.add((name, mod)) 
            elif x in [
                'ArrayLike',
                'MatrixLike',
                'FileLike',
                'PathLike',
                'Int',
                'Float',
                'Scalar',
                'Color'             
            ]:
                imports.add((x, typing))
        # Remove module qualifiers from qualnames now we have the import data we need
        components.add(qualname_re.sub(lambda m: m.group()[m.group().rfind('.')+1:], c))
            
    result = '|'.join(components)
    return result, imports
