from distutils import sysconfig;
import sys;

p = sysconfig.get_config_vars()['LIBDIR']
(M, m) = sys.version_info[:2]

print(f"{p}/libpython{M}.{m}.dylib")