# command to index cspad images

```
./msi.py  -j 10 -o $DD/cspadA/indexing -t cspad -glob "$DD/cspadA/job*/*npz" --basis-refine
```

# command to process indexing results

```
 srun -n10 -c1 libtbx.python process_mpi.py  --nrank 10 --ngpu 1 --glob "$DD/cspadA/indexing/job*/Els_cspad/El_*json" -o $DD/cspadA/agg
```


# command with difference expt and refls names

```
srun -n10 -c1 libtbx.python process_mpi.py  --sad --nrank 10 --ngpu 8 --glob "$DD/cssad/idx_out/idx-sad*refined.expt" -o $DD/cssad/agg --reflglob "$DD/cssad/idx_out/idx-sad_*strong.refl"
```
