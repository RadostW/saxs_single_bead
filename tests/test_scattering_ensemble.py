import pytest
import numpy as np
from saxs_single_bead.scattering_curve import scattering_curve
from saxs_single_bead.scattering_curve import scattering_curve_ensemble

def test_ensemble_average():
    sequence = "AAHH"
    locations_a = np.arange(3*len(sequence)).reshape(len(sequence),3)
    locations_b = np.arange(3*len(sequence)).reshape(len(sequence),3)
    locations = np.array([locations_a,locations_b])
    
    curves = np.array([
    scattering_curve( sequence , locations_a ),
    scattering_curve( sequence , locations_b )
    ])
    
    mean_curve = np.mean(curves,axis=0)
    
    ensemble_curve = scattering_curve_ensemble( sequence, locations)
    
    np.testing.assert_allclose(mean_curve,ensemble_curve)    
    

