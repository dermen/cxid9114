
from joblib import Parallel, delayed
import os
import h5py
import glob
import sys

n_jobs = 40  # gpu node
n_jobs = 32 # haswell

def main(jid, file_start):
    filedir = os.path.join(os.environ.get("DD") , "all_sims/agg" )
    files = glob.glob(filedir+ "/process*h5")

    jid = jid + file_start
    myfile = files[jid]
    with h5py.File(myfile, 'r') as h:
        n_shots =  h["h5_path"].shape[0]
    
    pkldir = os.environ.get("DD") + "/all_sims/timetest_haswell"
    #pkldir = os.environ.get("DD") + "/all_sims/timetest"
    #pkldir = os.environ.get("DD") + "/all_sims/pkl_mono_512"
    #pkldir = os.environ.get("DD") + "/all_sims/pkl_mono_6k"
    myoutdir = os.path.join(pkldir ,"results_job%d" % jid)
    if not os.path.exists(myoutdir):
        os.makedirs(myoutdir)

    mergepath = "/global/homes/d/dermen/cxid9114/sf/kaladin_allsims_2k_mark1.pkl"
    #mergepath = "/global/homes/d/dermen/cxid9114/sf/kaladin_2k_mark1.pkl"   
    #mergepath = "/global/homes/d/dermen/cxid9114/sf/kaladin_512_mark1.pkl"  
    #mergepath = "/global/homes/d/dermen/cxid9114/sf/kaladin_6k_mark1.pkl" 

    for i_shot in range(n_shots):
        outname = os.path.join(myoutdir, "job%d_shot%d" % (jid, i_shot))
        logname = os.path.join(myoutdir, "job%d_shot%d.log" % (jid, i_shot))
        if os.path.exists(outname + "_trial2.pkl"):
            print("%s exists!!!!!!!! continue" % outname)
            continue
        s='libtbx.python fat_data.py --glob %s --gainval 28 --sad --Ncells_size 13.7  --oversample 3 --bs7real --Fobs %s  --unknownscale 1e6 --verbose --scale 1 0 --ncells 1 0 --bg 0 0 --umatrix 0 1  --bmatrix 0 1  --fcell 0 0 --maxcalls 100 100 --ignorelinelow  --rotXYZsigma 0.001 0.001 0.001 --ucellsigma 0.1 --spotscalesigma 1 --ncellssigma 0.1 --loadstart %d --nload 1 --keeperstags stage1 stage2 --optoutname %s --tradeps=1 --Fref ../sf/bs7_real.pkl --forcemono --noprintresbins > %s'
        s = s % (myfile, mergepath, i_shot, outname, logname)
        os.system(s)

if __name__=="__main__":
    sel = int(sys.argv[1])
    if sel ==0:
        starts = [0]
        #starts = [0, 64, 128]
    else:
        #starts = [0+32, 64+32, 128+32]
        starts = [32]
    for start in starts:
        Parallel(n_jobs=n_jobs)(delayed(main)(jid, start) for jid in range(n_jobs))
