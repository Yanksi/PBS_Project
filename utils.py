import numpy as np

def readmesh():
    with open('bunny.off') as f:
        lines = f.readlines()
    Nv,Nf = int(lines[1].split()[0]) , int(lines[1].split()[1])
    V = []
    F = []
    Fid = []
    for i in range(Nv):
        v = [float(x) for x in lines[i+2].split()]
        V.append(v)

    scale = 0.15
    V = np.array(V)*scale
    t = np.array([10.0,10.0,-0.1])
    V = V + t

    for i in range(Nf):
        id = [int(x) for x in lines[i+2+Nv].split()]
        F.append( [ V[id[1]],V[id[2]],V[id[3]] ] )
        Fid.append(id[1:4])
    F = np.array(F)
    Fid = np.array(Fid)

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
    Fid = Fid.flatten()
    print(Fid.shape)
    
    return V.shape[0],F.shape[0],V,F,Fid

def readmesh2():
    V = [ [10.,0.,0.],[10.,20.,0.],[10.,0.,20.],[10.,20.,20.] ]
    Fid = [[0,1,2],[1,3,2]]

    F = []
    for face in Fid:
        F.append([ V[face[0]], V[face[1]], V[face[2]] ])
    
    V = np.array(V)
    F = np.array(F)
    Fid = np.array(Fid)
    Fid = Fid.flatten()
    return 4,2,V,F,Fid

def vec4(a1,a2,a3,a4):
    return np.array([a1,a2,a3,a4])

def mat4(a1,a2,a3,a4,b1,b2,b3,b4,c1,c2,c3,c4,d1,d2,d3,d4):
    return np.array([[a1,a2,a3,a4],[b1,b2,b3,b4],[c1,c2,c3,c4],[d1,d2,d3,d4]])

def bunnySDF(x,y,z):
    if (x*x+y*y+z*z > 1.):
        return np.sqrt(x*x+y*y+z*z)-.8

    f00=np.sin(y*vec4(-3.02,1.95,-3.42,-.60)+z*vec4(3.08,.85,-2.25,-.24)-x*vec4(-.29,1.16,-3.74,2.89)+vec4(-.71,4.50,-3.24,-3.50))
    f01=np.sin(y*vec4(-.40,-3.61,3.23,-.14)+z*vec4(-.36,3.64,-3.91,2.66)-x*vec4(2.90,-.54,-2.75,2.71)+vec4(7.02,-5.41,-1.12,-7.41))
    f02=np.sin(y*vec4(-1.77,-1.28,-4.29,-3.20)+z*vec4(-3.49,-2.81,-.64,2.79)-x*vec4(3.15,2.14,-3.85,1.83)+vec4(-2.07,4.49,5.33,-2.17))
    f03=np.sin(y*vec4(-.49,.68,3.05,.42)+z*vec4(-2.87,.78,3.78,-3.41)-x*vec4(-2.65,.33,.07,-.64)+vec4(-3.24,-5.90,1.14,-4.71))

    f10=np.sin(mat4(-.34,.06,-.59,-.76,.10,-.19,-.12,.44,.64,-.02,-.26,.15,-.16,.21,.91,.15)@f00+
        mat4(.01,.54,-.77,.11,.06,-.14,.43,.51,-.18,.08,.39,.20,.33,-.49,-.10,.19)@f01+
        mat4(.27,.22,.43,.53,.18,-.17,.23,-.64,-.14,.02,-.10,.16,-.13,-.06,-.04,-.36)@f02+
        mat4(-.13,.29,-.29,.08,1.13,.02,-.83,.32,-.32,.04,-.31,-.16,.14,-.03,-.20,.39)@f03+
        vec4(.73,-4.28,-1.56,-1.80))/1.0+f00
    f11=np.sin(mat4(-1.11,.55,-.12,-1.00,.16,.15,-.30,.31,-.01,.01,.31,-.42,-.29,.38,-.04,.71)@f00+
        mat4(.96,-.02,.86,.52,-.14,.60,.44,.43,.02,-.15,-.49,-.05,-.06,-.25,-.03,-.22)@f01+
        mat4(.52,.44,-.05,-.11,-.56,-.10,-.61,-.40,-.04,.55,.32,-.07,-.02,.28,.26,-.49)@f02+
        mat4(.02,-.32,.06,-.17,-.59,.00,-.24,.60,-.06,.13,-.21,-.27,-.12,-.14,.58,-.55)@f03+
        vec4(-2.24,-3.48,-.80,1.41))/1.0+f01
    f12=np.sin(mat4(.44,-.06,-.79,-.46,.05,-.60,.30,.36,.35,.12,.02,.12,.40,-.26,.63,-.21)@f00+
        mat4(-.48,.43,-.73,-.40,.11,-.01,.71,.05,-.25,.25,-.28,-.20,.32,-.02,-.84,.16)@f01+
        mat4(.39,-.07,.90,.36,-.38,-.27,-1.86,-.39,.48,-.20,-.05,.10,-.00,-.21,.29,.63)@f02+
        mat4(.46,-.32,.06,.09,.72,-.47,.81,.78,.90,.02,-.21,.08,-.16,.22,.32,-.13)@f03+
        vec4(3.38,1.20,.84,1.41))/1.0+f02
    f13=np.sin(mat4(-.41,-.24,-.71,-.25,-.24,-.75,-.09,.02,-.27,-.42,.02,.03,-.01,.51,-.12,-1.24)@f00+
        mat4(.64,.31,-1.36,.61,-.34,.11,.14,.79,.22,-.16,-.29,-.70,.02,-.37,.49,.39)@f01+
        mat4(.79,.47,.54,-.47,-1.13,-.35,-1.03,-.22,-.67,-.26,.10,.21,-.07,-.73,-.11,.72)@f02+
        mat4(.43,-.23,.13,.09,1.38,-.63,1.57,-.20,.39,-.14,.42,.13,-.57,-.08,-.21,.21)@f03+
        vec4(-.34,-3.28,.43,-.52))/1.0+f03

    f00=np.sin(mat4(-.72,.23,-.89,.52,.38,.19,-.16,-.88,.26,-.37,.09,.63,.29,-.72,.30,-.95)@f10+
        mat4(-.22,-.51,-.42,-.73,-.32,.00,-1.03,1.17,-.20,-.03,-.13,-.16,-.41,.09,.36,-.84)@f11+
        mat4(-.21,.01,.33,.47,.05,.20,-.44,-1.04,.13,.12,-.13,.31,.01,-.34,.41,-.34)@f12+
        mat4(-.13,-.06,-.39,-.22,.48,.25,.24,-.97,-.34,.14,.42,-.00,-.44,.05,.09,-.95)@f13+
        vec4(.48,.87,-.87,-2.06))/1.4+f10
    f01=np.sin(mat4(-.27,.29,-.21,.15,.34,-.23,.85,-.09,-1.15,-.24,-.05,-.25,-.12,-.73,-.17,-.37)@f10+
        mat4(-1.11,.35,-.93,-.06,-.79,-.03,-.46,-.37,.60,-.37,-.14,.45,-.03,-.21,.02,.59)@f11+
        mat4(-.92,-.17,-.58,-.18,.58,.60,.83,-1.04,-.80,-.16,.23,-.11,.08,.16,.76,.61)@f12+
        mat4(.29,.45,.30,.39,-.91,.66,-.35,-.35,.21,.16,-.54,-.63,1.10,-.38,.20,.15)@f13+
        vec4(-1.72,-.14,1.92,2.08))/1.4+f11
    f02=np.sin(mat4(1.00,.66,1.30,-.51,.88,.25,-.67,.03,-.68,-.08,-.12,-.14,.46,1.15,.38,-.10)@f10+
        mat4(.51,-.57,.41,-.09,.68,-.50,-.04,-1.01,.20,.44,-.60,.46,-.09,-.37,-1.30,.04)@f11+
        mat4(.14,.29,-.45,-.06,-.65,.33,-.37,-.95,.71,-.07,1.00,-.60,-1.68,-.20,-.00,-.70)@f12+
        mat4(-.31,.69,.56,.13,.95,.36,.56,.59,-.63,.52,-.30,.17,1.23,.72,.95,.75)@f13+
        vec4(-.90,-3.26,-.44,-3.11))/1.4+f12
    f03=np.sin(mat4(.51,-.98,-.28,.16,-.22,-.17,-1.03,.22,.70,-.15,.12,.43,.78,.67,-.85,-.25)@f10+
        mat4(.81,.60,-.89,.61,-1.03,-.33,.60,-.11,-.06,.01,-.02,-.44,.73,.69,1.02,.62)@f11+
        mat4(-.10,.52,.80,-.65,.40,-.75,.47,1.56,.03,.05,.08,.31,-.03,.22,-1.63,.07)@f12+
        mat4(-.18,-.07,-1.22,.48,-.01,.56,.07,.15,.24,.25,-.09,-.54,.23,-.08,.20,.36)@f13+
        vec4(-1.11,-4.28,1.02,-.23))/1.4+f13
    sdf = np.dot(f00,vec4(.09,.12,-.07,-.03))+np.dot(f01,vec4(-.04,.07,-.08,.05))+np.dot(f02,vec4(-.01,.06,-.02,.07))+np.dot(f03,vec4(-.05,.07,.03,.04))-0.16
    return sdf

def testsdf():
    SDF = []
    inside = []
    for i in range(21):
        for j in range(21):
            for k in range(21):
                x = i/10.0 -1.
                y = j/10.0 -1.
                z = k/10.0 -1.
                sdf = bunnySDF(x,y,z)
                if(sdf<0 and (x*x+y*y+z*z<1.) ):
                    inside.append([x,y,z])
                SDF.append(sdf)
    SDF = np.array(SDF)
    inside = np.array(inside)
    print( np.max(SDF),np.min(SDF) ) 
    print( np.sum(SDF<0)/11**3 )
    print(inside.shape)
    print( 'x: ',np.max(inside[:,0]),np.min(inside[:,0]) )
    print( 'y: ',np.max(inside[:,1]),np.min(inside[:,1]) )
    print( 'z: ',np.max(inside[:,2]),np.min(inside[:,2]) )
    return 0

def iniBall(pos,r):
    x=pos[0]
    y=pos[1]
    z=pos[2]
    ball = []
    dist = np.arange (-r, r+.1, .8)
    for i in dist:
        for j in dist:
            for k in dist:
                d = np.sqrt(i*i + j*j + k*k)
                if ( d<= r and  d>= r-2 ):
                    ball.append([i+x,j+y,k+z])
    ball = np.array(ball)
    sig = np.sum( np.var(ball,axis=0) )
    print(sig)

    return ball, sig


def iniBox(pos,l,w,h):
    x=pos[0]
    y=pos[1]
    z=pos[2]
    box=[]
    distl = np.arange(-l, l+.1, 1.)
    distw = np.arange(-w, w+.1, 1.)
    disth = np.arange(-h, h+.1, 1.)
    for i in distl:
        for j in distw:
            for k in disth:
                box.append([i+x,j+y,k+z])
    box = np.array(box)
    print(np.mean(box, axis=0))
    return box

def iniBall2(pos,r):
    x=pos[0]
    y=pos[1]
    z=pos[2]
    ball = []
    dist = np.arange (-r, r+.1, .5)
    for i in dist:
        for j in dist:
            for k in dist:
                d = np.sqrt(i*i + j*j + k*k)
                if (d<= r):
                    ball.append([i+x,j+y,k+z])
    ball = np.array(ball)
    print(np.sum(ball, axis=0))
    return ball

def iniBunny0():
    Bunny = []
    count = 0 
    dist = np.arange (-1., 1.001, .08)
    for x in dist:
        for y in dist:
            for z in dist:
                if(x*x+y*y+z*z<1):
                    sdf = bunnySDF(x,y,z)
                    if (sdf<0):
                        Bunny.append([x,y,z])
                        count += 1
    Bunny = np.array(Bunny)
    print(count)
    print(np.max(Bunny,axis=0),np.min(Bunny,axis=0), Bunny.shape)
    Bunny = 10*Bunny + np.array([10.,10.,10.])
    return Bunny

def iniBunny():
    vox = np.load('bunnyvoxel.npy')
    Bunny =[]
    print(vox.shape)
    count = 0 
    for i in range(0,vox.shape[0]):
        for j in range(0,vox.shape[0]):
            for k in range(0,vox.shape[0]):
                if(vox[i,j,k]<0):
                    count += 1
                    Bunny.append([float(i), float(j), float(k)])
    print(count)
    Bunny = np.array(Bunny)
    Bunny = Bunny*.8 + np.array([0.,0.,5.])
    Bunny2 = Bunny + np.array([0.,0.,30.])
    total = np.concatenate((Bunny,Bunny2), axis=0)
    print(Bunny.shape, np.max(Bunny,axis=0), np.min(Bunny,axis=0) )
    #return total
    return Bunny

testsdf()
#iniBall((5.,5.,5.),3.)
iniBox((5.,5.,5.),2.,2.,2.)
iniBunny()