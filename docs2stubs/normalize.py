import re


_restricted_val = re.compile(r'^(.*){(.*)}(.*)$')
_tuple1 = re.compile(r'^(.*)\((.*)\)(.*)$')  # using ()
_tuple2 = re.compile(r'^(.*)\[(.*)\](.*)$')  # using []
_sequence_of = re.compile(r'^(List|list|Sequence|sequence|Array|array) of ([A-Za-z0-9\._~`]+)$')
_tuple_of = re.compile(r'^(Tuple|tuple) of ([A-Za-z0-9\._~`]+)$')


def normalize_type(s: str) -> str:
    # Handle a restricted value set
    m = _restricted_val.match(s)
    l = None
    if m:
        s = m.group(1) + m.group(3)
        l = 'Literal[' + m.group(2) + ']'

    # Handle tuples in [] or (). Right now we can only handle one per line;
    # need to fix that.

    m = _tuple1.match(s)
    if not m:
        m = _tuple2.match(s)
    t = None
    if m:
        s = m.group(1) + m.group(3)
        t = 'tuple[' + m.group(2) + ']'

    # Now look at list of types. First replace ' or ' with a comma.
    # This is a bit dangerous as commas may exist elsewhere but 
    # until we find the failure cases we don't know how to address 
    # them yet.
    s = s.replace(' or ', ',')

    # Get the alternatives
    parts = s.split(',')

    def normalize_one(s):
        remap = {
            'array-like': 'ArrayLike',
            'callable': 'Callable',
        }
        """ Do some normalizing of a single type. """
        s = s.strip()
        s = s.replace('`', '')  # Removed restructured text junk

        # Handle collections like 'list of...', 'array of ...' ,etc
        m = _sequence_of.match(s)
        if m:
            return f'Sequence[{normalize_one(m.group(2))}]'
        m = _tuple_of.match(s)
        if m:
            return f'tuple[{normalize_one(m.group(2))}, ...]'

        # Handle literal numbers and strings
        if not (s.startswith('"') or s.startswith("'")):
            try:
                float(s)
            except ValueError:
                while s.startswith('.') or s.startswith('~'):
                    s = s[1:]

                if s in remap:
                    return remap[s]

                return s
        return 'Literal[' + s + ']'
        
    # Create a union from the normalized alternatives
    s = '|'.join(normalize_one(p) for p in parts if p.strip())

    # Add back our constrained value literal, if it exists
    if s and l:
        s += '|' + l
    elif l:
        s = l

    # Add back our tuple, if it exists
    if s and t:
        s += '|' + t
    elif t:
        s = t

    return s