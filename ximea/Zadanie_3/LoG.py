import numpy as np
import numba

@numba.njit
def _apply_raw_kernel_jit(_ker: numba.float64[:, :], _dat: numba.float64[:, :], 
                          _ker_shape: numba.types.UniTuple(numba.uint64, 2),  # type: ignore
                          _dat_shape: numba.types.UniTuple(numba.uint64, 2)): # type: ignore
    """Function used to make 2D convolution of _ker and _dat. 
    @numba.njit decorator used to compile this code runtime and speed up execution.
    _ker - 2D array of a convolution kernel
    _dat - 2D array of a data (an grey scale image)
    _ker_shape, _dat_shape - shapes of the corresponding arrays
    """

    # Usual 2D convolution code 
    # Visualization - https://en.wikipedia.org/wiki/Kernel_(image_processing)
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


def normalize(img):
    """Function for normalizing numpy array img. Works with any shape of the data.
    Normalizes values to fit in range [0,1]
    """
    _max = img.max()
    _min = img.min()
    return (img.astype(np.float64) + _min)/(_max - _min)



class Kernel:
    """Class Kernel provides abstraction for convolution Kernels.
    Main method is .apply_to(mat_like) - applies kernel to an image and returns resulting array.
    """
    def __init__(self, mat_like):
        """Constructor takes any object that can be iterated like 2D array. Data stored in plain style,
        Because of i tried to use native python arrays, but they don't work well with numba jit compilation.
        """
        self._raw = np.array(())
        row_len = len(mat_like[0])
        row_count = 0
        for row in mat_like:
            assert len(row) == row_len, "Error! Not homogeneous dimensions!"
            self._raw = np.append(self._raw, row)
            row_count += 1
        self._shape = (row_count, row_len)
    
    def apply_to(self, mat_like):
        """Convolves the kernel with mat_like object (an image). 
        I had to separate convolution code to use numba jit compilation.
        """
        _data = np.array(mat_like)
        _kernel = self._raw.reshape(self._shape)
        return _apply_raw_kernel_jit(_kernel, _data, _kernel.shape, _data.shape)
    
    @staticmethod
    def get_zero(m,n):
        """Simple method to get zero kernel
        """
        return Kernel([[0]*n]*m)
    
    @staticmethod
    def get_ones(m,n):
        """Simple method to get ones kernel
        """
        return Kernel([[1]*n]*m)
    
    def normalize(self):
        """Normalizes the kernel and returns self.
        """
        self._raw = self._raw/self._raw.sum()
        return self
    
    def __len__(self):
        """Amout of elements in the kernel
        """
        return self._shape[0]*self._shape[1]
    
    def __getitem__(self, *args):
         """Realisation of indexing for the class.
         in both k[i, j] and k[i][j] styles. (The second one is slower)
         """
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
        """Realisation of indexing for the class.
         in both k[i, j] and k[i][j] styles. (The second one is slower)
         """
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
        """Pretty looking string representing the kernel. For print statements.
        """
        l = []
        rc, _ = self._shape
        for i in range(rc):
            l.append(self[i].tolist())
        return f"Kernel({l})"


class Laplacian:
    """Abstraction layer for a laplacian Kernel. User has to set parameters and then call .build()
    to get an actual kernel. 
    """
    def __init__(self, signature=(1,1)):
        assert signature[0]!=0 and signature[1]!=0, "Error! Cannot use zero-signature!"
        self._m = signature[0]
        self._n = signature[1]

    def build(self):
        """Returns Laplacian kernel built using finite difference formula with step signature[0] for x axis
        and signature[1] for y axis. signature is from the constructor (self._m, self._n)
        """
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
    """Abstraction layer for gaussian Kernel. User has to set parameters and then call .build()
    to get the actual kernel. 
    """
    def __init__(self, shape=(5,5), variance=1):
        assert shape[0] > 0 and shape[1] > 0, "Error! Cannot use zero or negative shape!"
        self._shape = shape
        self._var = variance

    def build(self):
        """Returns guassian kernel with given in constructor shape and variance. The rule of 3*variance used to fit 
        gaussian distribution completely inside the kernel with variance=1 (default). Other values of variance (self._var)
        will scale the kernel relatively to the kernel grid.
        """
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
    """Abstraction layer for mean blur Kernel. User has to set parameters and then call .build()
    to get the actual kernel. 
    """
    def __init__(self, shape=(5,5)):
        assert shape[0] > 0 and shape[1] > 0, "Error! Cannot use zero or negative shape!"
        self._shape = shape

    def build(self):
        K = Kernel.get_ones(self._shape[0], self._shape[1])
        return K.normalize()
    
    def __repr__(self):
        return f"Mean({self._shape[0]},{self._shape[1]})"
    

