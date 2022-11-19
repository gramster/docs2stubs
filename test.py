import os
from typing_extensions import reveal_type


def test_normalize():
    from docs2stubs.type_normalizer import check_normalizer, load_map
    x = 'iterator over dict of str to any'
    print(check_normalizer(x, 'sklearn'))


def test_parser(): 
    from docs2stubs.docstring_parser import NumpyDocstringParser
    x = """
            Create legend handles and labels for a PathCollection.

        Each legend handle is a `.Line2D` representing the Path that was drawn,
        and each label is a string what each Path represents.

        This is useful for obtaining a legend for a `~.Axes.scatter` plot;
        e.g.::

            scatter = plt.scatter([1, 2, 3],  [4, 5, 6],  c=[7, 2, 3])
            plt.legend(*scatter.legend_elements())

        creates three legend elements, one for each color with the numerical
        values passed to *c* as the labels.

        Also see the :ref:`automatedlegendcreation` example.

        Parameters
        ----------
        prop : {"colors", "sizes"}, default: "colors"
            If "colors", the legend handles will show the different colors of
            the collection. If "sizes", the legend will show the different
            sizes. To set both, use *kwargs* to directly edit the `.Line2D`
            properties.
        num : int, None, "auto" (default), array-like, or `~.ticker.Locator`
            Target number of elements to create.
            If None, use all unique elements of the mappable array. If an
            integer, target to use *num* elements in the normed range.
            If *"auto"*, try to determine which option better suits the nature
            of the data.
            The number of created elements may slightly deviate from *num* due
            to a `~.ticker.Locator` being used to find useful locations.
            If a list or array, use exactly those elements for the legend.
            Finally, a `~.ticker.Locator` can be provided.
        fmt : str, `~matplotlib.ticker.Formatter`, or None (default)
            The format or formatter to use for the labels. If a string must be
            a valid input for a `.StrMethodFormatter`. If None (the default),
            use a `.ScalarFormatter`.
        func : function, default: ``lambda x: x``
            Function to calculate the labels.  Often the size (or color)
            argument to `~.Axes.scatter` will have been pre-processed by the
            user using a function ``s = f(x)`` to make the markers visible;
            e.g. ``size = np.log10(x)``.  Providing the inverse of this
            function here allows that pre-processing to be inverted, so that
            the legend labels have the correct values; e.g. ``func = lambda
            x: 10**x``.
        **kwargs
            Allowed keyword arguments are *color* and *size*. E.g. it may be
            useful to set the color of the markers if *prop="sizes"* is used;
            similarly to set the size of the markers if *prop="colors"* is
            used. Any further parameters are passed onto the `.Line2D`
            instance. This may be useful to e.g. specify a different
            *markeredgecolor* or *alpha* for the legend handles.

        Returns
        -------
        handles : list of `.Line2D`
            Visual representation of each element of the legend.
        labels : list of str
            The string labels for elements of the legend.
    """

        
    rtn = NumpyDocstringParser().parse(x)
    for i, k in enumerate(['Params', 'Returns', 'Attrs']):
        sec = rtn[i]
        if sec:
            print(k)
            print('-' * len(k))
            for (n, r, t) in sec:
                print(f'  name {n}: raw {r}, normalized {t}')


def test_analyzer(m: str = 'matplotlib'):
    from docs2stubs import analyze_module
    analyze_module(m)


def test_stubber(m: str = 'matplotlib'):
    from docs2stubs import stub_module
    stub_module(m)


def test_get_package_files(m = 'matplotlib'):
    from docs2stubs.utils import get_module_and_children
    modules = [m]
    while modules:
        m = modules.pop()
        mod, file, submodules = get_module_and_children(m)
        if not mod or not file:
            continue
        modules.extend(submodules)
        print(f'\n{m}\n{"="*len(m)}')
        f = file
        i = f.find('/site-packages/')
        if i > 0:
            f = f[i+15:]
        print(f)


if __name__ == '__main__':
    #test_analyzer('sklearn')
    test_normalize()
    #test_get_package_files()
    #test_stubber('sklearn')
    #from docs2stubs.normalize import _is_string
    #print(_is_string("'balanced'"))
