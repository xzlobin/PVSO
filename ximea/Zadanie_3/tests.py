from LoG import Kernel

import unittest

class TestKernel(unittest.TestCase):

    def setup(self):
        self.kernel = Kernel([[0,0,0]
                              [0,1,0]
                              [0,0,0]])

if __name__ == '__main__':
    unittest.main()