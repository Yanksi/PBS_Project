import numpy as np

def readmesh():
    with open('bunny.off') as f:
        lines = f.readlines()
    Nv,Nf = int(lines[1].split()[0]) , int(lines[1].split()[1])
    V = []
    F = []
    for i in range(Nv):
        v = [float(x) for x in lines[i+2].split()]
        V.append(v)

    scale = 0.12
    V = np.array(V)*scale
    t = np.array([10.0,10.0,0.0])
    V = V + t

    for i in range(Nf):
        id = [int(x) for x in lines[i+2+Nv].split()]
        F.append( [ V[id[0]],V[id[1]],V[id[2]] ] )
    F = np.array(F)

    Pn = []
    Pd = []
    for i in range(Nf):
        p = F[i,0]
        q = F[i,1]
        r = F[i,2]
        v1 = p - q
        v2 = p - r
        v = np.cross(v1,v2)
        v = v / np.linalg.norm(v)
        d = np.dot(p,v)
        Pn.append(v)
        Pd.append(d)
    Pn = np.array(Pn)
    Pd = np.array(Pd)

    print(Nv,Nf,F.shape, V.shape, Pn.shape, Pd.shape)
    print(np.max(V[:,0]), np.min(V[:,0]))
    print(np.max(V[:,1]), np.min(V[:,1]))
    print(np.max(V[:,2]), np.min(V[:,2]))
    return V,F

readmesh()