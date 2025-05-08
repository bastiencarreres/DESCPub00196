import os
import glob
import copy
import snanapytools as snt
import pandas as pd
from astropy.io import fits, ascii
import numpy as np
from astropy.table import Table
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(
                    prog='Apply Select',
                    description='Apply a selection to SNANA SIM / FIT files')

parser.add_argument('pip_dir')           
parser.add_argument('out_dir')     
parser.add_argument('-sf', '--suffix', default='fake_select')
parser.add_argument('-sx', '--sigmoid_x', default=19.8)
parser.add_argument('-sscale', '--sigmoid_scale', default=0.15)
parser.add_argument('-sb', '--sigmoid_bands', nargs='*', default=['r'])
parser.add_argument('-only_inc', '--only_include',nargs='*', default=[])
parser.add_argument('-exc', '--exclude',nargs='*', default=[])
parser.add_argument('-ovw', '--overwrite', action='store_true')
parser.add_argument('-rs', '--random_seed', type=int, default=123456789)

args = parser.parse_args()

def P_det(x):
    return  1/(1 + np.exp((x - args.sigmoid_x) / args.sigmoid_scale))

selec = {'bands': args.sigmoid_bands, 'prob': P_det}

snt.tools.apply_selec_pippin_dir(args.pip_dir, args.out_dir, selec, 
                                 suffix=args.suffix,  
                                 overwrite=args.overwrite,
                                 exclude=args.exclude,     
                                 only_include=args.only_include, 
                                 random_seed=args.random_seed)
