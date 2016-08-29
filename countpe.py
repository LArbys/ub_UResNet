import numpy as np
import ROOT as rt
from larcv import larcv

def countpe( ioman, pmtproducer ):
    peimg = ioman.get_data( larcv.kProductImage2D, pmtproducer )
    pmtwfms = peimg.Image2DArray()[0]
    pmtnd = larcv.as_ndarray( pmtwfms )
    # slice off beam window
    beamwin = pmtnd[:,190:310]
    pedwin  = pmtnd[:,0:120]
    # remove pedestal
    beamwin -= (2048) # 0.5 pe-ish threshold
    pedwin  -= (2048)
    # remove undershooot
    beamwin[ beamwin<0 ] = 0.0
    pedwin[ beamwin<0 ] = 0.0
    # sum pe
    chsum = np.sum( beamwin, axis=1 )
    chsum /= 100.0
    maxch = np.argmax( chsum )
    totsum = np.sum( chsum )
    pedsum = np.sum( np.sum( pedwin, axis=1 )/100.0 )
    
    return totsum, maxch, pedsum
