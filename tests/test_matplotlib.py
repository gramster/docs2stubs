import pytest
from docs2stubs.type_normalizer import check_normalizer
from hamcrest import assert_that, equal_to


def gentest(input, modname, __):
    trivial, type, imports = check_normalizer(input, modname)
    print(f'   {"" if trivial else "n"}tcheck("{input}", "{type}", {imports})')

