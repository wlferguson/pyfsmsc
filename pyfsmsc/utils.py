import requests 
from collections.abc import Iterable 

import numpy as np

def hello(name):
    print(f'Welcome to pyfsmsc: {name}')

def RDF_to_SQ(r, gr, density, qmin, qmax, nqs):
    rs = r;
    dr = r[1] - r[0];
    
    RDFs = gr - 1

    qs = np.linspace(qmin, qmax, num = nqs)
    Sqs = np.zeros(qs.shape[0])

    for qind, q in enumerate(qs):
        s = 0
        for rind, r in enumerate(rs):
            s += dr * r * np.sin(q * r) * RDFs[rind]
            Sqs[qind] = 1 + 4 * np.pi * density * s / q
            
    return qs, Sqs
