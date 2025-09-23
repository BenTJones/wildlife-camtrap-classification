import os, sys, importlib.util
print('cwd =', os.getcwd())
print('exe =', sys.executable)
print('spec =', importlib.util.find_spec('src.data.make_manifest'))
