#/bin/sh

find . -name .ipynb_checkpoints | xargs rm -rf
find . -name __pycache__ | xargs rm -rf
rm -rf build/
rm -rf src/gkvfigpdf.egg-info/
rm -rf examples/figpdf_*
rm -f tests/log_BZX.dat tests/metric_boozer.bin.dat
rm -rf .pytest_cache/
