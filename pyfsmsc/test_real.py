import pytest
import pyfsmsc
import numpy as np
import pyfsmsc.reciprocal

@pytest.fixture()
class ReciprocalFTSQ:
    data = "examples/grSample.txt"
    rGr = np.loadtxt(data, usecols = [0,-1])

    def test_FT_SQ(self, rGr):

        self.rGr = rGr

        density = 160000/54.6399**3*0.5
        qmin = 2; qmax = 10
        nqs = 1000

        assert 1 == 2

        #q, Sq = pyfsmsc.real.RDF_to_SQ(self.rGr[:,0],self.rGr[:,1], density, qmin, qmax, nqs)
        #assert np.max(Sq) == 3