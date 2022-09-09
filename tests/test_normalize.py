import pytest
from docs2stubs.normalize import check_normalizer
from hamcrest import assert_that, equal_to


def test_simple_normalizations():
    for typ in ['int', 'float', 'None']:
      assert_that(check_normalizer(typ), equal_to(f'(Trivial) {typ}'))


