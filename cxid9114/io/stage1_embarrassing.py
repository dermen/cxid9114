
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--aggdir", type=str,required=True, help="path to folder containing diffBragg file")
parser.add_argument("--mergepkl", type=str, required=True, help="path to merge pickle file")
parser.add_argument("--outdir", type=str, required=True, help="path to output folder where pandas dataframes will be stored" )
parser.add_argument("--initNcells", type=float, required=True, help="initial mosaic parameter m")
parser.add_argument("--readoutless", action="store_true")
parser.add_argument("--j", type=int, default=None)
args = parser.parse_args()

from joblib import Parallel, delayed, effective_n_jobs
import os
import h5py
import glob
import sys

FILES = glob.glob(os.path.join( args.aggdir, "process*h5"))
if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

def main(jid):

    jid = jid 
    myfile = FILES[jid]
    with h5py.File(myfile, 'r') as h:
        n_shots =  h["h5_path"].shape[0]
    
    myoutdir = os.path.join(args.outdir ,"results_job%d" % jid)
    if not os.path.exists(myoutdir):
        os.makedirs(myoutdir)

    for i_shot in range(n_shots):
        outname = os.path.join(myoutdir, "job%d_shot%d" % (jid, i_shot))
        logname = os.path.join(myoutdir, "job%d_shot%d.log" % (jid, i_shot))
        #if os.path.exists(outname + "_trial2.pkl"):
        #    print("%s exists!!!!!!!! continue" % outname)
        #    continue
        
        output_cmd = "> %s" % logname
        if jid==0:
            output_cmd = "" #| tee %s" % logname
        s='libtbx.python fat_data.py --glob %s --gainval 28 --sad --Ncells_size %f  --oversample 3 --bs7real --Fobs %s  --unknownscale 1e6 --verbose --scale 1 0 --ncells 1 0 --bg 0 0 --umatrix 0 1  --bmatrix 0 1  --fcell 0 0 --maxcalls 100 100 --ignorelinelow  --rotXYZsigma 0.001 0.001 0.001 --ucellsigma 0.1 --spotscalesigma 1 --ncellssigma 0.1 --loadstart %d --nload 1 --keeperstags stage1 stage2 --optoutname %s --tradeps=1 --forcemono --noprintresbins %s %s'
        readoutless = ""
        if args.readoutless:
            readoutless = "--readoutless"
        s = s % (myfile, args.initNcells,args.mergepkl, i_shot, outname,readoutless, output_cmd)
        if jid==0:
            print(s)
        os.system(s)

n = effective_n_jobs()
if len(FILES) > n:
    print("WARNING: Trying to process %d files with %d processors" % (len(FILES),n))

n_jobs = len(FILES)

if args.j is not None:
    n_jobs = args.j

Parallel(n_jobs=n_jobs)(delayed(main)(jid) for jid in range(n_jobs))

