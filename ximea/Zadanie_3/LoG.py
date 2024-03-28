import numpy
import copy
import numba
from numba.experimental import jitclass
spec = [
    ('_shape',  numba.types.UniTuple(numba.float64, 2) ),
    ('_raw',    numba.float64[:]), 
    ('_output', numba.float64[:])
]

#@jitclass(spec)
class Kernel:
    def __init__(self, mat_like):
        self._raw = numpy.array(())
        row_len = len(mat_like[0])
        row_count = 0
        for row in mat_like:
            assert len(row) == row_len, "Error! Not homogeneous dimensions!"
            #self._raw.extend(row)
            self._raw = numpy.append(self._raw, row)
            row_count += 1
        self._shape = (row_count, row_len)

    def apply_to(self, mat_like):
        self._output = numpy.copy(mat_like)
        n = self._output.shape[0]
        m = self._output.shape[1]
        for i in range(n):
            for j in range(m):
                self._output[i][j] = self._apply_inplace(mat_like, anchor=(i,j))

        return self._output
    
    @staticmethod
    def get_zero(m,n):
        return Kernel([[0]*n]*m)
    
    def _apply_inplace(self, mat_like, anchor=(0,0)):
        rc, rl = self._shape
        c_i = rc // 2
        c_j = rl // 2
        _buffer = 0
        for i in range(rc):
            for j in range(rl):
                ii = anchor[0] - c_i + i
                jj = anchor[1] - c_j + j
                if ii < 0 or jj < 0 or ii >= len(mat_like) or jj >= len(mat_like[0]): continue
                _buffer += self[i,j]*mat_like[ii][jj]

        return _buffer
    
    def __len__(self):
        return self._shape[0]*self._shape[1]
    
    def __getitem__(self, *args):
         rc, rl = self._shape

         # kernel[i,j]
         if isinstance(args[0], tuple):
            i, j = args[0][0], args[0][1]
            if i < 0: i = rc + i
            if j < 0: j = rl + j
            idx_l = rl*i + j
            if j >= rl and i >= rc: raise IndexError
            return self._raw[idx_l]
         
         # kernel[i] row like
         elif isinstance(args[0], int):
            i = args[0]
            if i < 0: i = rc + i
            if i >= rc: raise IndexError
            return self._raw[(rl*i):(rl*i+rl)]
    
    def __setitem__(self, key, value):
        rc, rl = self._shape

        # kernel[i,j]
        if isinstance(key, tuple):
            i, j = key[0], key[1]
            if i < 0: i = rc + i
            if j < 0: j = rl + j
            idx_l = rl*i + j
            if j >= rl and i >= rc: raise IndexError
            self._raw[idx_l] = value
         
        # kernel[i] row like
        elif isinstance(key, int):
            i = key
            if i < 0: i = rc + i
            if i >= rc: raise IndexError
            self._raw[(rl*i):(rl*i+rl)] = value

         
#    def __repr__(self):
#        l = []
#        rc, _ = self._shape
#        for i in range(rc):
#            l.append(self[i].tolist())
#        return f"Kernel({l})"


class Laplacian:
    def __init__(self, signature=(1,1)):
        assert signature[0]!=0 and signature[1]!=0, "Error! Cannot use zero-signature!"
        self._m = signature[0]
        self._n = signature[1]

    def build(self):
        m, n = (self._m, self._n)
        K = Kernel.get_zero(2*m+1, 2*n+1)
        m_mult = 1/m**2
        n_mult = 1/n**2
        K[m,n] = -2*m_mult - 2*n_mult
        K[0,n] = K[-1, n] = n_mult
        K[m,0] = K[m, -1] = m_mult
        return K
    
    def __repr__(self):
        return f"Laplacian({self._m},{self._n})"
         
#import numpy as np
#mat = np.array([[1,2],[3,4]])
#ker_test = np.array([[1,2,3],[4,5,6],[7,8,9]])
#k = Kernel(ker_test)
#res = k.apply_to(mat)
#print(res)
