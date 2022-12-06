import abc
import collections
import re
from typing import Any, Callable
from .utils import Sections


class Deque(collections.deque):
    """
    A subclass of deque that adds `.Deque.get` and `.Deque.next` methods.
    """

    sentinel = object()

    def get(self, n: int) -> Any:
        """
        Return the nth element of the stack, or ``self.sentinel`` if n is
        greater than the stack size.
        """
        return self[n] if n < len(self) else self.sentinel

    def next(self) -> Any:
        if self:
            return super().popleft()
        else:
            raise StopIteration


class DocstringParserBase(abc.ABC):
    """ Methods that are the same in Napoleon for Google format
        and Numpydoc format I put in this class. That will make 
        it easier eventually to add Google format support if desired.
    """
    _single_colon_regex = re.compile(r'(?<!:):(?!:)')
    _xref_or_code_regex = re.compile(
        r'((?::(?:[a-zA-Z0-9]+[\-_+:.])*[a-zA-Z0-9]+:`.+?`)|'
        r'(?:``.+?``))')
    _remove_default_val = re.compile(r'^(.*),[ \t]*default[ \t]*.*$')
    _remove_optional = re.compile(r'^(.*),[ \t]*[Oo]ptional$')

    def __init__(self):
        self._attributes = None
        self._parameters = None
        self._returns = None
        self._is_in_section = False
        self._section_indent = 0
        self._lines: Deque = Deque()
        self._sections: dict[str, Callable] = {}

    @abc.abstractmethod
    def _is_section_break(self) -> bool:
            ...

    @abc.abstractmethod
    def _is_section_header(self) -> bool:
        ...

    def _is_indented(self, line: str, indent: int = 1) -> bool:
        """ Check if a line is at least <indent> indented """
        for i, s in enumerate(line):
            if i >= indent:
                return True
            elif not s.isspace():
                return False
        return False

    def _get_indent(self, line: str) -> int:
        """ Get indentation for a single line. """
        for i, s in enumerate(line):
            if not s.isspace():
                return i
        return len(line)

    def _get_current_indent(self, peek_ahead: int = 0) -> int:
        line = self._lines.get(peek_ahead)
        while line is not self._lines.sentinel:
            if line:
                return self._get_indent(line)
            peek_ahead += 1
            line = self._lines.get(peek_ahead)
        return 0

    def _consume_empty(self) -> None:
        """ Advance through any empty lines. """
        line = self._lines.get(0)
        while self._lines and not line:
            self._lines.next()
            line = self._lines.get(0)

    def _consume_indented_block(self, indent: int = 1) -> None:
        line = self._lines.get(0)
        while (
            not self._is_section_break() and (not line or self._is_indented(line, indent))
        ):
            self._lines.next()
            line = self._lines.get(0)

    def _consume_to_next_section(self) -> None:
        """ Consume a whole section. """
        self._consume_empty()
        while not self._is_section_break():
            self._lines.next()

    def _consume_section_header(self) -> str:
        section = self._lines.next()
        stripped_section = section.strip().strip(':')
        if stripped_section.lower() in self._sections:
            section = stripped_section
            self._lines.next() # consume ----- part
        return section

    def _skip_section(self, section: str):
        self._consume_to_next_section()

    def _partition_field_on_colon(self, line: str) -> tuple[str, str, str]:
        before_colon = []
        after_colon = []
        colon = ''
        found_colon = False
        for i, source in enumerate(DocstringParserBase._xref_or_code_regex.split(line)):
            if found_colon:
                after_colon.append(source)
            else:
                m = DocstringParserBase._single_colon_regex.search(source)
                if (i % 2) == 0 and m:
                    found_colon = True
                    colon = source[m.start(): m.end()]
                    before_colon.append(source[:m.start()])
                    after_colon.append(source[m.end():])
                else:
                    before_colon.append(source)

        return ("".join(before_colon).strip(),
                colon,
                "".join(after_colon).strip())


    @abc.abstractmethod
    def _consume_field(self, prefer_type: bool = False
                       ) -> tuple[str, str]: ...

    def _consume_fields(self, parse_type: bool = True, prefer_type: bool = False,
                        multiple: bool = False) -> dict[str, str]:
        self._consume_empty()
        fields = {}
        while not self._is_section_break():
            name, typ = self._consume_field(prefer_type)

            # Remove , optional ... from end
            m = DocstringParserBase._remove_optional.match(typ)
            if m:
                typ = m.group(1)
            # Remove , default ... from end
            m = DocstringParserBase._remove_default_val.match(typ)
            if m:
                typ = m.group(1)

            # Remove (default) from within
            typ = typ.replace(' (default)', '')

            if multiple and name:
                for n in name.split(","):
                    fields[n.strip()] = typ
            elif name or typ:
                fields[name] = typ
        return fields

    @abc.abstractmethod
    def _parse_returns_section(self, section: str) -> None:
        ...

    def _parse_attributes_section(self, section: str) -> None:
        self._attributes = self._consume_fields(multiple=True)

    def _parse_parameters_section(self, section: str) -> None:
        self._parameters = self._consume_fields(multiple=True)

    def _prep_parser(self, docstring: str) -> None:
        self._attributes = None
        self._parameters = None
        self._returns = None
        self._is_in_section = False
        self._section_indent = 0
        self._lines = Deque(map(str.rstrip, docstring.splitlines()))

    def parse(self, docstring: str) -> Sections:
        self._prep_parser(docstring)
        self._consume_to_next_section()
        while self._lines:
            section = self._consume_section_header()
            if not section:
                # IMO this shouldn't happen but does; dig into it
                # later
                self._consume_to_next_section()
                continue

            self._is_in_section = True
            self._section_indent = self._get_current_indent()
            self._sections[section.lower()](section)
            self._is_in_section = False
            self._section_indent = 0

        return Sections(params=self._parameters, 
                        returns=self._returns,
                        attrs=self._attributes)


class NumpyDocstringParser(DocstringParserBase):

    _numpy_section_regex = re.compile(r'^[=\-`:\'"~^_*+#<>]{2,}\s*$')

    def __init__(self): 
        super().__init__()
        self._sections: dict[str, Callable] = {
            'attributes': self._parse_attributes_section,
            'examples': self._skip_section,
            'methods': self._skip_section,
            'notes': self._skip_section,
            'other parameters': self._skip_section,
            'parameters': self._parse_parameters_section,
            'receives': self._skip_section,
            'returns': self._parse_returns_section,
            'raises': self._skip_section,
            'references': self._skip_section,
            'see also': self._skip_section,
            'warnings': self._skip_section,
            'warns': self._skip_section,
            'yields': self._skip_section,
        }

    def _is_section_header(self) -> bool:
        section, underline = self._lines.get(0), self._lines.get(1)
        section = section.strip().lower()
        if section in self._sections:
            if isinstance(underline, str):
                return bool(NumpyDocstringParser._numpy_section_regex.match(underline.strip()))

        return False

    def _is_section_break(self) -> bool:
        line1, line2 = self._lines.get(0), self._lines.get(1)
        return (not self._lines or
                self._is_section_header() or
                ['', ''] == [line1, line2] or
                (self._is_in_section and
                    line1 and
                    not self._is_indented(line1, self._section_indent)))

    def _consume_field(self, prefer_type: bool = False
                       ) -> tuple[str, str]:
        line = self._lines.next()
        
        _name, _, _type = self._partition_field_on_colon(line)
        _name, _type = _name.strip(), _type.strip()

        if prefer_type and not _type:
            _type, _name = _name, _type

        # Consume the description. Sometimes this will contain
        # type information too, but there's no way to tell; safer
        # to just discard it.
        self._consume_indented_block(self._get_indent(line) + 1)
        return _name, _type

    def _parse_returns_section(self, section: str) -> None:
        self._returns = self._consume_fields(prefer_type=True)

