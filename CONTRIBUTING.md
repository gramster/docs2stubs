## Building

1) Install flit of needed: pip -m install flit
2) Run 'flit install'

You should then have a local copy available as 'docs2stubs'.

## Releasing

To make a release,

  1) Update README.md and the \__version__ in docs2stubs/\__init__.py
  2) Run 'flit install'
  3) Test the installed docs2stubs locally
  4) Upload to PyPI: 'flit publish'

