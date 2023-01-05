#find submodules/scipy -type d -name "tests" -exec cp -Rp {} scipy-tests/{} \;
(cd submodules/scipy; find . -path "*/tests/*" | cpio -pmud ../../scipy-tests)
 

