# coding: utf-8
from scipy.stats import pearsonr, spearmanr
from pylab import *
import tabulate


def cc_merge(merge_even_f, merge_odd_f, channel=0, nbins=15, a=78.97, c=38.12, dmin=2.1):
    
    h0,I0,err0, mult0 = np.load(merge_even_f, allow_pickle=True)['merge%d' % channel].T
    h1,I1,err1, mult1 = np.load(merge_odd_f, allow_pickle=True)['merge%d' % channel].T
    h0 = list(map(tuple,h0))
    h1 = list(map(tuple,h1))
    H0 = {h:v for h,v in zip(h0,I0)}
    H1 = {h:v for h,v in zip(h1,I1)}
    hcommon = set(H0).intersection(H1)
    
    all_res, all_val0, all_val1 = [],[],[]
    for hasu in hcommon:
        h,k,l = hasu
        val0 = H0[hasu]
        val1 = H1[hasu]
        q = np.sqrt( h**2/a**2 + k**2/a**2 + l**2/c**2)
        res = 1/q
        all_val0.append(val0)
        all_val1.append(val1)
        all_res.append(res)
        
    res,val0,val1= map(np.array, [all_res, all_val0, all_val1] )
    order = np.argsort(res)
    res,val0,val1 = map(lambda x: x[order], [res,val0,val1])
    res_for_bins = res[res >= dmin]
    bins = [r[0] for r in np.array_split(res_for_bins, nbins)] + [res_for_bins[-1]]
    digs = np.digitize(res,bins)-1
    
    out = []
    for i_bin in range(len(bins)-1):
        sel = digs==i_bin
        res_bin = res[sel]
        val0_bin = val0[sel]
        val1_bin = val1[sel]
        cchalf = pearsonr(val0_bin, val1_bin)[0]
        ccstar = 0
        if cchalf > 0:
            ccstar = np.sqrt(2*cchalf/(1+cchalf))
        out.append((res_bin.mean(), cchalf, ccstar, res_bin.min(), res_bin.max()))

    dspace,cchalf, ccstar, dmin, dmax = zip(*out)

    overall_cchalf = pearsonr(val0, val1)[0]
    overall_cc = [overall_cchalf, 0 if overall_cchalf < 0 else np.sqrt(2*overall_cchalf/(1+overall_cchalf))]

    return dspace, cchalf, ccstar, dmin, dmax, overall_cc


def plot_cc(dsp, cchalf, ccstar, figname=None):
    plot( dsp, cchalf, marker='o', label="CC1/2", ms=10, lw=2)
    plot( dsp, ccstar, marker='*', label="CC*", ms=10, lw=2, color='tomato')
    gca().grid(1)
    gca().invert_xaxis()
    gca().tick_params(labelsize=18)
    ylabel("CC", fontsize=18)
    xlabel("d-spacing ($\AA$)", fontsize=20)
    legend(prop={"size":16})
    subplots_adjust(bottom=0.17, left=0.15, top=0.95, right=0.95)

    if figname is not None:
        savefig(figname)
        close()
    else:
        show()


def get_table(dmin, dmax, cchalf, ccstar):
    order = np.argsort(dmin)[::-1]
    dmin, dmax, cchalf, ccstar = map(lambda x: np.array(x)[order], (dmin, dmax, cchalf, ccstar))
    table = tabulate.tabulate(
        list(zip(dmax, dmin, cchalf, ccstar)), 
        tablefmt='simple', 
        headers=["dmax","dmin","CC1/2","CC*"], 
        floatfmt=(".3f", ".3f", ".2f", ".2f"))
    return table


if __name__=="__main__":
    mf1 = "merge_m1.npz"
    mf2 = "merge_m2.npz"
    dsp, cchalf, ccstar, dmin, dmax = cc_merge(mf1, mf2)
    print( get_table(dmin, dmax, cchalf, ccstar))
    plot_cc(dsp, cchalf, ccstar, "cc_merge_test.png")
    plot_cc(dsp, cchalf, ccstar, )
