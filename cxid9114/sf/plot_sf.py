
import numpy as np

from cxid9114.sf import struct_fact_special
data = np.load("scanned_fp_fdp.npz")
x,fp,fdp = data["ev_range"], data["fp"], data["fdp"]

Yb = struct_fact_special.Yb_scatter()
fp_henke = [Yb.fp_tbl(ev=e) for e in x]
fdp_henke = [Yb.fdp_tbl(ev=e) for e in x]

# sasaki tables
#fp_sas = [Yb.fp_tbl(ev=e, henke=False) for e in x]
#fdp_sas = [Yb.fdp_tbl(ev=e, henke=False) for e in x]

from pylab import *

figure(figsize=(3.5,5.5))
ax = gca()
plot( x, fp_henke, '--', color="C0")
plot( x, fp, color="steelblue")
plot( x, fdp_henke, '--',color="C3")
plot( x, fdp,color="tomato")

xlabel("photon energy (eV)", fontsize=12)
ax.tick_params(labelsize=11)
ylabel("$e^{-1}$", fontsize=12)
ax.grid(1, alpha=0.5)
xlim(x[0], x[-1])
ax.legend(("$f\,'$ henke", "$f\,'$", "$f\,'' \
    henke$","$f\,''$"), prop={"size":11}, loc=1)
subplots_adjust(left=.2, bottom=.15, right=.95, top=.95)
show()

