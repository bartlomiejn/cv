from distutils import sysconfig
import sys

libpath = sysconfig.get_config_vars()['LIBDIR']
(M, m) = sys.version_info[:2]

print(f'{libpath}/libpython{M}.{m}.dylib')