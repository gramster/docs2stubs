#!/bin/bash
#
# Do some post-processing of networkx

for i in `find typings/networkx -name '*.pyi'`
do
    sed -i '' 's/import networkx as nx/from networkx.classes.graph import Graph/' $i
    sed -i '' 's/(G,/(G:Graph,/' $i
    sed -i '' 's/ G,/ G:Graph,/' $i
    sed -i '' 's/(G)/(G:Graph)/' $i
    sed -i '' 's/@not_implemented_for.*$//' $i
    sed -i '' 's/@open_file.*$//' $i
    sed -i '' 's/^def _.*$//' $i
done
for i in `echo typings/networkx/*/*/*.pyi`
do
    sed -i '' 's/^from networkx\./from .../' $i
done
for i in `echo typings/networkx/*/*.pyi`
do
    sed -i '' 's/^from networkx\./from ../' $i
done
for i in `echo typings/networkx/*.pyi`
do
    sed -i '' 's/^from networkx\./from ./' $i
done
for i in `echo typings/networkx/*.pyi`
do
    sed -i '' 's/^from networkx /from . /' $i
done

