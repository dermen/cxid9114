from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--prefix", help="output file prefix", type=str, required=True)
args = parser.parse_args()

from cxid9114.sf.struct_fact_special import sfgen
from cxid9114 import utils                                                                                       
from cxid9114.parameters import WAVELEN_HIGH


bs7_all = sfgen(wavelen_A=WAVELEN_HIGH, pdb_name="../sim/4bs7.pdb", 
    algo='fft', dmin=1.5, ano_flag=True, yb_scatter_name='scanned_fp_fdp.tsv', only_yb=False)                                                 
bs7_for_CCano = sfgen(wavelen_A=WAVELEN_HIGH, pdb_name="../sim/4bs7.pdb", 
    algo='fft', dmin=1.5, ano_flag=True, yb_scatter_name='scanned_fp_fdp.tsv', only_yb=True)                                            
bs7_all = bs7_all.as_amplitude_array()                                                                          
bs7_for_CCano = bs7_for_CCano.as_amplitude_array()                                                              

out_F_real = "%s_real.pkl" % args.prefix
out_F_for_CCano = "%s_CCano.pkl" % args.prefix
utils.save_flex(bs7_all, out_F_real)                                                                    
utils.save_flex(bs7_for_CCano, out_F_for_CCano) 
print("Wrote |F| reference %s" % out_F_real)
print("Wrote |F| reference with only Yb undergoing anomalous scattering %s" % out_F_for_CCano)

