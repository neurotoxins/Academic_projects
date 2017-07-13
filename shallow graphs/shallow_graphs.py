
gtext = sc.textFile("a2/graphs/Assign2_100.txt").map(lambda s: s.split(" ")).cache()

(m,n)  = gtext.map(lambda s: (int(s[0]), int(s[1]))).reduce(lambda a,b: map(max, zip(a,b)))

assert m==n

block_size = 50

def fill_block(B, x):
    B[x[0]%block_size, x[1]%block_size] = x[2]
    return B

A = gtext.map(lambda s: ( (int(s[0])/block_size, (int(s[1]))/block_size), (int(s[0]), int(s[1]), float(s[2])))) \
    .aggregateByKey(np.matrix(np.zeros((block_size, block_size))), fill_block, add).cache()

def is_full(mat):
    return np.count_nonzero(mat) == np.prod(mat.shape)

# checking if A is full ...
A.values().map(is_full).reduce(lambda x,y: x & y)
Ap1 = A.map(addI)
A1 = A.map(lambda (k,v): (k[1], (k[0],v)))

AsqpA = A1.join(Ap1).map(lambda (k,v): ((v[0][0], v[1][0]), v[0][1]*v[1][1])) \
    .reduceByKey(add)


# test for A shallowness
AsqpA.values().map(is_full).reduce(lambda x,y: x&y)

G = load_matrix("a2/graphs/Assign2_100.txt", (100, 100))


g2p1 = G*G + G
is_full(g2p1)
