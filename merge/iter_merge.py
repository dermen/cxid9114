import os

idir = "$CSCRATCH/kaladin/"
cmd = "srun -n 40 -c 2 merge.sh 0 %s" % idir
print(cmd)
os.system(cmd)
for i in range(1,10):
    mtz_prev = "../MERGE/%03d/out/%03d_all.mtz" % (i-1, i-1)
    cmd = "srun -n 40 -c2 merge_mark0.sh %d %s %s" % (i,idir,  mtz_prev)
    print(cmd)
    os.system(cmd)
