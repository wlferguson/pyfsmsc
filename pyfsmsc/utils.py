import requests 
from collections.abc import Iterable 

def hello(name):
    print(f'Hey there {name}')

def RDF_to_SQ(r, gr, density, qmin, qmax, nqs):
    rs = r;
    dr = r[1] - r[0];
    
    #RDFs = gr - np.ones_like(rs)

    #qs = np.linspace(qmin, qmax, num = nqs)
    #Sqs = np.zeros_like(qs)

    #for qind, q in enumerate(qs):
    #    s = 0
    #    for rind, r in enumerate(rs):
    #    s += dr * r * np.sin(q * r) * RDFs[rind]
    #    Sqs[qind] = 1 + 4 * np.pi * density * s / q  
    return dr



    
    
    
    