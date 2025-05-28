# gkvfigpdf - Generating PDF of GKV standard output
---

**gkvfigpdf** is a Python package that generates a PDF of GKV standard ASCII output.
(Previous `fig_stdout` tool using shell/awk/gnuplot/latex is converted to Python.)


## Installation

To install from PyPI:
```bash
pip install gkvfigpdf
```
Or install the latest development version from GitHub:
```bash
pip install git+https://github.com/GKV-developers/gkvfigpdf.git
```


## Usage

#### **(i) Basic usage: As a command line tool**
```sh
python -m gkvfigpdf -d DIR
```
The argument `DIR` is the GKV output directory. The namelist file `DIR/gkvp.namelist.001`, log file `DIR/log/gkvp.000000.0.log.001`, and hst directory `DIR/hst/` should exist.
You get a summary PDF file `CWD/figpdf_yyyymmdd_hhmmss/fig_stdout.pdf`.

#### **(ii) As a Python function**

```python
from gkvfigpdf import gkvfigpdf

gkvfigpdf(gkv_stdout_dir="YOUR GKV OUTPUT DIR")
```
You get a summary PDF file `CWD/figpdf_yyyymmdd_hhmmss/fig_stdout.pdf`, always in the current working directory `CWD`.


## Dependencies

gkvfigpdf requires the following Python packages:
- `numpy`, `matplotlib`, `pandas`, `reportlab`, `pypdf`

## License

This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for details.

## Author

Developed by Shinya Maeyama (maeyama.shinya@nifs.ac.jp)
