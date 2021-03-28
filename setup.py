from distutils.core import setup, Extension
import numpy as np
from sysconfig import get_paths

#print("Python include paths : ", get_paths()['include'])
#print("Numpy include paths : ", np.get_include())

module1 = Extension('TemporalStability',
                    include_dirs=[get_paths()['include'],
                                  np.get_include()],
                    #libraries = ['python'],
                    library_dirs=[get_paths()['stdlib']],
                    sources=['TS/TemporalStability.cpp', 
                              'TS/RunLengthEncoding.cpp'],
                    language='c++')

setup(name='Temporal Stability',
      version='1.0',
      description='Temporal Stability function warpping package',
      ext_modules=[module1])
