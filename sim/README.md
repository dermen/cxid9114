### Simulation command to get realistic noise data

```
srun --pty ./d9114_sims.py  -o datasf2 -odir paint --gpu --add-bg --add-noise --profile gauss --bg-name boop_cspad.h5 --cspad -g 1  -trials 10 --show-params --oversample 0 --Ncells 30 --xtal_size_mm 0.00015 --mos_doms 1 --write-img  --readout  --datasf --overwrite --masterscalejitter 0.1 --Ncellsjitter 3 --masterscale .34
```

### Another simulation command that uses structure factors deposited in PDB for 4bs7

```
srun --pty   ./d9114_sims.py  -o realer -odir paint/ --gpu --add-bg --add-noise --profile gauss --bg-name boop_cspad.h5 --cspad -g 1  -trials 10 --show-params --oversample 0 --Ncells 30 --xtal_size_mm 0.00015 --mos_doms 1 --write-img  --readout  --datasf --overwrite --masterscalejitter 0.2 --Ncellsjitter 3 --masterscale 0.7 --kernelspergpu 1
```
