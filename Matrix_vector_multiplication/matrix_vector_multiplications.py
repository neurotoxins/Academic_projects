from operator import add
import numpy as np
import numpy.random

atext = sc.textFile("a2/mats/a_100x200.txt").map(lambda s: s.split(" ")).cache()
btext = sc.textFile("a2/mats/b_200x100.txt").map(lambda s: s.split(" ")).cache()


def fill_block(B, x):
    B[x[0]%block_size, x[1]%block_size] = x[2]
    return B

def load_matrix(fname, sz):
    M = np.matrix(np.zeros(sz))
    with open(fname) as f:
        content = f.readlines()
        for l in content:
            s = l.split()
            M[int(s[0]), int(s[1])] = float(s[2])
    return M


# lets get the dimensions of the matrices
# assume matrices are mxn and nxp
m = atext.map(lambda s: int(s[0])).reduce(max)

# lets do better and do it in one sweep.
(m,n)  = atext.map(lambda s: (int(s[0]), int(s[1]))) \
    .reduce(lambda a,b: map(max, zip(a,b)))
(n1,p) = btext.map(lambda s: (int(s[0]), int(s[1]))) \
    .reduce(lambda a,b: map(max, zip(a,b)))

assert n == n1

m = m+1
n = n+1
p = p+1

print m, n, p

# Testing two pass approach first
x = sc.parallelize([((0,0),'a'), ((0,1),'b'), ((1,0),'c'), ((1,1),'d')])
y = sc.parallelize([((0,0),'e'), ((0,1),'f'), ((1,0),'g'), ((1,1),'h')])

# j, ('M', i, m_ij)
x1 = x.map(lambda (k,v): (k[1], (k[0],v)))
# j, ('N', k, n_jk)
y1 = y.map(lambda (k,v): (k[0], (k[1],v)))

# now we gather by the common key j
z = x1.join(y1).map(lambda (k,v): ((v[0][0], v[1][0]), v[0][1]+v[1][1])) \
    .reduceByKey(lambda a,b: '+'.join([a,b]))
sorted(z.collect())



A = load_matrix("a2/mats/a_100x200.txt", (100, 200))
B = load_matrix("a2/mats/b_200x100.txt", (200, 100))
C = A*B

block_size = 2



a = atext.map( lambda s:((int(s[0])/block_size, (int(s[1]))/block_size), (int(s[0]), int(s[1]), float(s[2])))) \
    .aggregateByKey(np.matrix(np.zeros((block_size, block_size))), fill_block, add).cache()

b = btext.map(lambda s: ( (int(s[0])/block_size, (int(s[1]))/block_size), (int(s[0]), int(s[1]), float(s[2])))) \
    .aggregateByKey(np.matrix(np.zeros((block_size, block_size))), fill_block, add).cache()


a1 = a.map(lambda (k,v): (k[1], (k[0],v)))
b1 = b.map(lambda (k,v): (k[0], (k[1],v)))


c = a1.join(b1).map(lambda (k,v): ((v[0][0], v[1][0]), v[0][1]*v[1][1])) \
    .reduceByKey(add)
sorted(c.collect())
