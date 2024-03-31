import numpy as np
import numba

@numba.njit
def _apply_raw_kernel_jit(_ker: numba.float64[:, :], _dat: numba.float64[:, :], 
                          _ker_shape: numba.types.UniTuple(numba.uint64, 2),  # type: ignore
                          _dat_shape: numba.types.UniTuple(numba.uint64, 2)): # type: ignore
    out = np.zeros(_dat_shape, dtype=np.float64)
    c_k = _ker_shape[0] // 2
    c_l = _ker_shape[1] // 2
    for i in range(_dat_shape[0]):
            for j in range(_dat_shape[1]):
                out[i, j] = 0.0
                for l in range(_ker_shape[0]):
                    for k in range(_ker_shape[1]):
                        ii = i - c_k + k
                        jj = j - c_l + l
                        if ii < 0 or jj < 0 or ii >= _dat_shape[0] or jj >= _dat_shape[1]: continue
                        out[i, j] += _ker[l,k]*_dat[ii, jj]
    return out


class Kernel:
    def __init__(self, mat_like):
        self._raw = np.array(())
        row_len = len(mat_like[0])
        row_count = 0
        for row in mat_like:
            assert len(row) == row_len, "Error! Not homogeneous dimensions!"
            self._raw = np.append(self._raw, row)
            row_count += 1
        self._shape = (row_count, row_len)

    def apply_to(self, mat_like):
        self._output = np.copy(mat_like)
        shp = np.shape(self._output)
        for i in range(shp[0]):
            for j in range(shp[1]):
                self._output[i, j] = self._apply_inplace(mat_like, anchor=(i,j))

        return self._output
    
    def jit_apply_to(self, mat_like):
        _data = np.array(mat_like)
        _kernel = self._raw.reshape(self._shape)
        return _apply_raw_kernel_jit(_kernel, _data, _kernel.shape, _data.shape)
    
    @staticmethod
    def get_zero(m,n):
        return Kernel([[0]*n]*m)
    
    @staticmethod
    def get_ones(m,n):
        return Kernel([[1]*n]*m)
    
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
    
    def normalize(self):
        self._raw = self._raw/self._raw.sum()
        return self
    
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

         
    def __repr__(self):
        l = []
        rc, _ = self._shape
        for i in range(rc):
            l.append(self[i].tolist())
        return f"Kernel({l})"


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
    
class Gaussian:
    def __init__(self, shape=(5,5), variance=1):
        assert shape[0] > 0 and shape[1] > 0, "Error! Cannot use zero or negative shape!"
        self._shape = shape
        self._var = variance

    def build(self):
        K = Kernel.get_zero(self._shape[0], self._shape[1])
        c_i = self._shape[0] // 2
        c_j = self._shape[1] // 2

        for i in range(self._shape[0]):
            for j in range(self._shape[1]):
                x = 3*2*(i - c_i)/self._shape[0]
                y = 3*2*(j - c_j)/self._shape[1]
                K[i,j]=np.exp(-(x**2+y**2)/(2*self._var**2))

        return K.normalize()
    
    def __repr__(self):
        return f"Gaussian({self._shape[0]},{self._shape[1]} | variance={self._var})"
    
class Mean:
    def __init__(self, shape=(5,5)):
        assert shape[0] > 0 and shape[1] > 0, "Error! Cannot use zero or negative shape!"
        self._shape = shape

    def build(self):
        K = Kernel.get_ones(self._shape[0], self._shape[1])
        return K.normalize()
    
    def __repr__(self):
        return f"Mean({self._shape[0]},{self._shape[1]})"
    

