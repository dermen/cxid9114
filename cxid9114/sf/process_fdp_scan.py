"""
RUN THIS SCRIPT AND IT WILL WALK YOU THROUGH THE PROCESS
OF INTEPOLATING THE SCAN DATA TO GET "F DBL PRIME" AND THEN 
INTEGRATING TO GET "F PRIME"
"""
# coding: utf-8

# NOTE: scan data is in pixels,
# and by measuring scale bars we get
# the following
ev_perpix = 100 / 222.8
electron_ppix = 10/148.58

# load in the data which is stored in csv
import numpy as np
d = np.loadtxt("fdp_scan.csv", delimiter=',')
x,y = d.T  # this is the eV(s) and electron(s) axis 


from pylab import *
plot( x,y,)
title("Note: electron scale is inverted..")
xlabel("scanned pixels")
ylabel("scanned pixels")
show()

# we know at what point electrons is 0
yo = 2126  # read off of the scan
yscaled = (y-yo)*electron_ppix*-1  # -1 is to invert the axis again
xlabel("scanned pixels")
ylabel("e-")
plot( x, yscaled)   # a plot of pixels versus electrons
title("Got the y-axis scaled to electrons!")
show()

# we know the peak in x-axis corresonds  to 8944
xmax = x[np.argmax(yscaled)]
xscaled = (x-xmax)*ev_perpix + 8944


plot( xscaled, yscaled, label="data")
xlabel("eV")
ylabel("e-")
title("Got the x-axis scaled to electron volts!")

# inteprolate the data now
from scipy.interpolate import interp1d
I_fdp = interp1d( xscaled, yscaled, kind='linear', 
        bounds_error=False, fill_value=0)
ev_range = np.arange( 8850, 9080,0.2 )
plot( ev_range, I_fdp(ev_range) ,'--',label="interpolated")

show()

# PAD WITH  zeros for integration
# e- decay with increasing energy
# (slope taken from henke table after LIII (8944) edge)
m_e = -0.0019672131147541  

# interpolate the scanned data to 0 so that we can integrate to infinity
outer_range = np.arange(9080, 15000,0.2)
inner_range = np.arange(5000, 8850, 0.2)

# fit polynomial to smooth out decays a bit
Iouter = lambda x: (x-9080)*m_e + I_fdp(9080)[()]
Youter = [i if i > 0 else 0 for i in Iouter(outer_range)]
Youter_smooth = polyval(polyfit(outer_range, Youter,deg=13), outer_range)

Iinner = lambda x: (-x+8850)*m_e + I_fdp(8850)[()]
Yinner = [i if i > 0 else 0 for i in Iinner(inner_range)]
Yinner_smooth = polyval(polyfit(inner_range, Yinner,deg=13), inner_range)

# concatenate all arrays for use in integra;l
all_x = hstack((inner_range, ev_range, outer_range ))
all_y = hstack((Yinner_smooth, I_fdp(ev_range), Youter_smooth))

#TODO : use FFT convolution tricks as described in SHERRELL thesis
plot( all_x, all_y, label='f dbl prime' )
ax = gca()
ev_running = []
fp_integ = []
xlabel("eV")
ylabel("e-")
title("results of f prime  integral")
vlines(8944,-50,60, color='#777777',alpha=0.7, lw=2, label='LIII edge')
for i_E, E in enumerate(ev_range):
    dE = all_x[1] - all_x[0]
    
    e = all_x
    Fdp = all_y
    vals = Fdp *dE* e / (E**2 - e**2) 
    #vals = [Fdp *dE* e / (E**2 - e**2) for e,Fdp in zip(all_x, all_y)]
    # NOTE: clean up the slops
    vals = np.array(vals)
    vals[isinf(vals)] = 0
    vals = np.nan_to_num(vals)   # tears 
    
    result = vals.sum()
    fp_integ.append( result)
    ev_running.append(E)
    if i_E % 20==0:
        print ("Integral at %.3f (%d / %d) value = %.3f " % (E,i_E, len(ev_range), result))
        if len(ax.lines)==1:
            plot(  ev_running, fp_integ, label="f prime")
            legend()
            xlim(8800, 9100)
        else:
            ax.lines[1].set_data( (ev_running, fp_integ))
        

        draw()
        pause(0.1)

show()

# SAVE the results!
#np.savez("scanned_fp_fdp", 
#    ev_range=ev_range, 
#    fp=fp_integ, 
#    fdp=I_fdp(ev_range))
#

