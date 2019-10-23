### Command to filter cspad bboxes:

```
srun -n1 -c1 --pty libtbx.python filter_bboxes.py   --glob "../indexing/agg/process_rank0.h5" --reslow 4 --reshigh 2.5 --snrmin 5
```


### Test command

```
libtbx.python load_bboxes.py  --glob process_rank_cspad.h5 --verbose --testmode
```

(Requires the [diffBragg](https://github.com/cctbx/cctbx_project/tree/diffBragg) branch of ```cctbx_project```)

### Results 

On mac using commit e5105baecd3def9fd9ab0dcba9d6d19d6f83273e convergence was achieved in 49 iterations

```
Compute functional and gradients Iter 49 (Using Curvatures)
<><><><><><><><><><><><><>
Running diffBragg over spot 24/24 on panel 15  LBFGS stp: residual=-3971.8276, ncells=29.421892, detdist=-123.92715, gain=  1, scale=1.4512602
Ucell: a_Ang=+78.95894, c_Ang=+38.12056 *** Missets: rotX=+0.0009307935, rotY=-0.0006173365, rotZ=+0.0003708


LBFGS GRADS: |x|=488.799401, |g|=1.110388, Gncells=-0.000044, Gdetdist=  0, Ggain=  0, Gscale=-0.00015859554
GUcell: Ga_Ang=+0.01249168, Gc_Ang=+0.004766684 *** GMissets: GrotX=-1.092435, GrotY=-0.09595159, GrotZ=-0.1737


Rank 0, filename=cspad_rank0_data25.h5, ang=0.002358, init_ang=0.050015, a=78.958943, init_a=79.155637, c=38.120564, init_c=38.160762
```
