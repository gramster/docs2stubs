# Utilities for working with MonkeyType traces.
# Borrows heavily from the MonkeyType code. However, as
# I want to incorporate this into the docs2stubs CST tree
# walker, I want to be able to access the traced types directly.

from typing import IO, Optional, Tuple, List, Dict, Any, Iterable, cast, Type

# A variant of the SQL store that uses counts instead of timestamps.
# This is duplicated in the tracing directory, so need to refactor
# at some point to avoid that.

import datetime
import os
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


def get_toplevel_function_signature(module: str, function: str):
    stubs = _get_module_stubs(_dbpath, _pkg, module)
    if stubs:
        try:
            return stubs.function_stubs[function].signature
        except:
            return None
    return None

def get_method_signature(module: str, class_: str, method: str):
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
