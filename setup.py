# coding:utf-8
import os

from distutils.core import setup
from Cython.Build import cythonize

'''
TODO:
    使用Cpython 编译python文件，将.py文件编译成.c文件和.so文件
USAGE:
    python setup.py build_ext --inplace
    python3 setup.py build_ext
'''
# 在列表中输入需要加密的py文件
key_funs = [
    'server/tasks.py',
    'server/models.py',
    'server/sse.py',
    'server/app.py',
]
# 遍历需要加密的py文件

setup(
    name="know", 
    ext_modules=cythonize(key_funs),
)

print('Done!')

