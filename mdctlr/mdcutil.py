# MDC repo utility function
import os
from os.path import exists, join
import sys
from scipy.io import loadmat
import numpy as np
import json
import sys
from os import path
import re

def snr(xtrue : np.ndarray, xapprox : np.ndarray):
    """calculate signal noise ratio of two vectors."""
    return - 20 * np.log10(np.linalg.norm(xapprox-xtrue) / np.linalg.norm(xtrue))

def createfolder(foldername):
    if not exists(foldername):
        os.mkdir(foldername)

##################################################################
# conver python file to jupyter notebook
##################################################################
def py2nb(py_str):
    cells = []
    stext = py_str.split('\n')

    splitidx = []
    typeinfo = []
    stextnew = []
    for idx,x in enumerate(stext):
        if x.startswith('# %% [markdown]'):
            splitidx.append(idx)
            typeinfo.append('markdown')
        elif x.startswith('# %%'):
            splitidx.append(idx)
            typeinfo.append('code')
    splitidx.append(len(stext))
    print(splitidx)
    for idx,curinfo in enumerate(splitidx[:-1]):
        cell_type = typeinfo[idx]
        if cell_type == "markdown":
            chunktext = []
            for x in stext[curinfo+1 : splitidx[idx+1]]:
                chunktext.append( x[2:] )
        else:
            chunktext = []
            for x in stext[curinfo+1 : splitidx[idx+1]]:
                chunktext.append(x)
        cell = {'cell_type': cell_type,'metadata': {},
            'source': '\n'.join(chunktext).splitlines(True),}
        if cell_type == 'code':
            cell.update({'outputs': [], 'execution_count': None})
        cells.append(cell)
    
    notebook = {
        'cells': cells,
        'metadata': {
            'anaconda-cloud': {},
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3'},
            'language_info': {
                'codemirror_mode': {'name': 'ipython', 'version': 3},
                'file_extension': '.py',
                'mimetype': 'text/x-python',
                'name': 'python',
                'nbconvert_exporter': 'python',
                'pygments_lexer': 'ipython3',
                'version': '3.6.1'}},
        'nbformat': 4,
        'nbformat_minor': 4
    }

    return notebook


def convert(in_file, out_file):
    _, in_ext = path.splitext(in_file)
    _, out_ext = path.splitext(out_file)

    assert(in_ext == '.py' and out_ext == '.ipynb')

    with open(in_file, 'r', encoding='utf-8') as f:
        py_str = f.read()
    notebook =  py2nb(py_str)
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)

def checkmpi4py():
    import pkg_resources
    installed = {pkg.key for pkg in pkg_resources.working_set}
    return 'mpi4py' in installed