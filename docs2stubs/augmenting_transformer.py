import os
from pathlib import Path
import sys
from typing import IO, Sequence

import sqlite3

import libcst as cst
from libcst import parse_module
from libcst.codemod import CodemodContext
from libcst.codemod.visitors import ApplyTypeAnnotationsVisitor
from libcst.helpers import get_full_name_for_node
from libcst.codemod.visitors._apply_type_annotations import FunctionKey
from libcst._nodes.internal import CodegenState

from monkeytype.stubs import build_module_stubs_from_traces
from monkeytype.exceptions import MonkeyTypeError
from monkeytype.typing import NoOpRewriter
from monkeytype.stubs import (
    ExistingAnnotationStrategy,
    Stub,
    build_module_stubs_from_traces,
)
from monkeytype.cli import display_sample_count
from monkeytype.db.base import CallTraceStore, CallTraceThunk
from monkeytype.encoding import CallTraceRow, serialize_traces
from monkeytype.config import DefaultConfig
from monkeytype.db.sqlite import SQLiteStore

from .utils import Sections, get_module_and_children, load_fullmap, save_result

# A variant of the SQL store that uses counts instead of timestamps.


#DEFAULT_TABLE = "monkeytype_call_traces"
QueryValue = str|int
ParameterizedQuery = tuple[str, list[QueryValue]]


class SQLiteDedupStore(SQLiteStore):

    @classmethod
    def make_store(cls, connection_string: str) -> "CallTraceStore":
        conn = sqlite3.connect(connection_string)
        return cls(conn)

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self.table = "monkeytype_call_traces"

    def make_query(self, table: str, module: str, qualname: str|None, limit: int) -> ParameterizedQuery:
        raw_query = """
        SELECT
            module, qualname, arg_types, return_type, yield_type
        FROM {table}
        WHERE
            module == ?
        """.format(
            table=table
        )
        values: list[QueryValue] = [module]
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

    def filter(
        self, module: str, qualname_prefix: str|None = None, limit: int = 2000
    ) -> list[CallTraceThunk]:
        sql_query, values = self.make_query(self.table, module, qualname_prefix, limit)
        with self.conn:
            cur = self.conn.cursor()
            cur.execute(sql_query, values)
            return [CallTraceRow(*row) for row in cur.fetchall()]

        
class MyConfig(DefaultConfig):

    def __init__(self, db: str) -> None:
        self._db = db

    def trace_store(self):
        return SQLiteDedupStore.make_store(self._db)


def get_annotation(n: cst.CSTNode|None):
    if n is None:
        return None
    state = CodegenState(default_indent='', default_newline='')
    n._codegen_impl(state, default_indicator='')  # type: ignore
    annotation = (''.join(state.tokens)).strip().\
            replace('List', 'list').\
            replace('Dict', 'dict').\
            replace('ndarray', 'NDArray').\
            replace('Tuple', 'tuple')
    addnone = False
    if annotation.startswith('Optional['):
        addnone = True
        annotation = annotation[9:-1]
    if annotation.startswith('Union['):
        if annotation[6:-1].find('[') < 0:
            annotation = annotation[6:-1]
            annotation = '|'.join(annotation.split(','))
    if addnone:
        annotation += '|None'

    return annotation


class MyApplyTypeAnnotationsVisitor(ApplyTypeAnnotationsVisitor):

    """ This is a kludge at class level but I didn't want to mess with __init__ """
    fullmap: Sections|None = None

    def _update_parameters(
        self,
        fkey: str,
        annotations,
        updated_node: cst.FunctionDef,
    ) -> cst.Parameters:
        # Update params and default params with annotations
        # Don't override existing annotations or default values unless asked
        # to overwrite existing annotations.
        def update_annotation(
            fkey: str,
            parameters: Sequence[cst.Param],
            annotations: Sequence[cst.Param],
            positional: bool,
        ) -> list[cst.Param]:
            parameter_annotations = {}
            annotated_parameters = []
            positional = positional and not self.strict_posargs_matching
            for i, parameter in enumerate(annotations):
                key = i if positional else parameter.name.value
                if parameter.annotation:
                    parameter_annotations[key] = parameter.annotation.with_changes(
                        whitespace_before_indicator=cst.SimpleWhitespace(value="")
                    )
            for i, parameter in enumerate(parameters):
                key = i if positional else parameter.name.value
                if key in parameter_annotations:
                    old_annotation = get_annotation(parameter.annotation)
                    new_annotation = get_annotation(parameter_annotations[key])
                    overwrite = self.overwrite_existing_annotations or not parameter.annotation
                    if old_annotation is None or old_annotation == 'Any':
                        overwrite = True
                    elif old_annotation != new_annotation:
                        print(f'Function {fkey} Parameter {key}: old: {old_annotation}, new: {new_annotation}')

                    if overwrite:
                        parameter = self._apply_annotation_to_parameter(
                            parameter=parameter,
                            annotation=parameter_annotations[key],
                        )
                annotated_parameters.append(parameter)
            return annotated_parameters

        return updated_node.params.with_changes(
            params=update_annotation(
                fkey,
                updated_node.params.params,
                annotations.parameters.params,
                positional=True,
            ),
            kwonly_params=update_annotation(
                fkey,
                updated_node.params.kwonly_params,
                annotations.parameters.kwonly_params,
                positional=False,
            ),
            posonly_params=update_annotation(
                fkey,
                updated_node.params.posonly_params,
                annotations.parameters.posonly_params,
                positional=True,
            ),
        )

    def leave_FunctionDef(
        self,
        original_node: cst.FunctionDef,
        updated_node: cst.FunctionDef,
    ) -> cst.FunctionDef:
        key = FunctionKey.make(self._qualifier_name(), updated_node.params)
        self.qualifier.pop()
        if key in self.annotations.functions:
            function_annotation = self.annotations.functions[key]
            # Only add new annotation if:
            # * we have matching function signatures and
            # * we are explicitly told to overwrite existing annotations or
            # * there is no existing annotation
            if not self._match_signatures(updated_node, function_annotation):
                return updated_node

            set_return_annotation = (
                self.overwrite_existing_annotations or updated_node.returns is None
            )

            if (updated_node.returns is not None and function_annotation.returns is not None):
                old_annotation = get_annotation(updated_node.returns)
                new_annotation = get_annotation(function_annotation.returns)
                if (new_annotation != old_annotation):
                    if old_annotation == 'Any':
                        set_return_annotation = True
                    else:
                        print(f'{key.name}: old: {old_annotation}, new: {new_annotation}')

            if set_return_annotation and function_annotation.returns is not None:
                updated_node = self._apply_annotation_to_return(
                    function_def=updated_node,
                    annotation=function_annotation.returns,
                )
            # Don't override default values when annotating functions
            new_parameters = self._update_parameters(key.name, function_annotation, updated_node)
            return updated_node.with_changes(params=new_parameters)
        return updated_node

        
    def leave_Assign(
        self,
        original_node: cst.Assign,
        updated_node: cst.Assign,
    ) -> cst.Assign|cst.AnnAssign:

        self.current_assign = None

        if len(original_node.targets) > 1:
            for assign in original_node.targets:
                target = assign.target
                if isinstance(target, (cst.Name, cst.Attribute)):
                    name = get_full_name_for_node(target)
                    if name is not None and name != "_":
                        # Add separate top-level annotations for `a = b = 1`
                        # as `a: int` and `b: int`.
                        self._add_to_toplevel_annotations(name)
            return updated_node
        else:
            return self._annotate_single_target(original_node, updated_node)
        
    pass


def get_stub(
    module, config: MyConfig, stdout: IO[str], stderr: IO[str], 
    existing_annotation_strategy: ExistingAnnotationStrategy = ExistingAnnotationStrategy.IGNORE,
      verbose: bool = False,
    limit: int = 2000, disable_type_rewriting: bool = False, sample_count: bool = False) -> Stub|None:

    thunks = config.trace_store().filter(module, None, limit)
    traces = []
    failed_to_decode_count = 0
    for thunk in thunks:
        try:
            traces.append(thunk.to_trace())
        except MonkeyTypeError as mte:
            if verbose:
                print(f"WARNING: Failed decoding trace: {mte}", file=stderr)
            failed_to_decode_count += 1
    if failed_to_decode_count and not verbose:
        print(
            f"{failed_to_decode_count} traces failed to decode; use -v for details",
            file=stderr,
        )
    if not traces:
        return None
    rewriter = config.type_rewriter()
    if disable_type_rewriting:
        rewriter = NoOpRewriter()
    stubs = build_module_stubs_from_traces(
        traces,
        config.max_typed_dict_size(),
        existing_annotation_strategy=existing_annotation_strategy,
        rewriter=rewriter,
    )
    if sample_count:
        display_sample_count(traces, stderr)
    return stubs.get(module, None)


def apply_stub_using_libcst(
    module: str, stub: str, source: str, overwrite_existing_annotations: bool,
) -> str:
    try:
        stub_module = parse_module(stub)
        source_module = parse_module(source)
        context = CodemodContext()
        MyApplyTypeAnnotationsVisitor.store_stub_in_context(
            context,
            stub_module,
            overwrite_existing_annotations,
        )
        transformer = MyApplyTypeAnnotationsVisitor(context)
        transformed_source_module = transformer.transform_module(source_module)
    except Exception as exception:
        print(f"{module}: Failed applying stub with libcst: {exception}")
        return source
    return transformed_source_module.code


def complain_about_no_traces(module, stderr: IO) -> None:
    # When there is no trace and a top level module's filename is passed, print
    # a useful error message.
    if os.path.exists(module):
        print(f"No traces found for {module}; did you pass a filename instead of a module name? "
              f"Maybe try just '{os.path.splitext(module)[0]}'.", file=stderr)
    else:
        print(f'No traces found for module {module}', file=stderr)


def apply_stub_handler(
    module: str,
    source_path: str,
    config: MyConfig,
    existing_annotation_strategy: ExistingAnnotationStrategy = ExistingAnnotationStrategy.OMIT,
    stdout: IO[str] = sys.stdout, stderr: IO[str] = sys.stderr
) -> None:
    stub = get_stub(module, config, stdout, stderr)
    if stub is None:
        complain_about_no_traces(module, stderr)
        return
    source_with_types = apply_stub_using_libcst(
        module,
        stub=stub.render(),
        source=Path(source_path).read_text(),
        overwrite_existing_annotations=existing_annotation_strategy
        == ExistingAnnotationStrategy.IGNORE,
    )
    Path(source_path).write_text(source_with_types)
    #print(source_with_types, file=stdout)


def augment_module(m: str, include_submodules: bool = True, stub_folder: str = 'typings', trace_folder: str = '.') -> None|tuple:
    config = MyConfig(f"{trace_folder}/{m}.sqlite3")
    orig_m = m

    MyApplyTypeAnnotationsVisitor.fullmap = load_fullmap('analysis', m)

    modules = [m]
    while modules:
        m = modules.pop()
        mod, file, submodules = get_module_and_children(m)
        if include_submodules:
            if not mod:
                continue
            modules.extend(submodules)
        else:
            if not mod:
                return

        result = None

        if file is None:
            continue

        try:
            with open(file) as f:
                source = f.read()
        except Exception as e:
            print(f"Failed to read {file}: {e}")
            continue

        source_path = stub_folder + file[file.rfind('/site-packages/') + 14:] + 'i'
        apply_stub_handler(m, source_path, config)
        print(f"Processed file {file}")
