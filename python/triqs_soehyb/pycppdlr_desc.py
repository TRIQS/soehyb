
# Poor man's wrapper for cppdlr::imtime_ops
# Author: Hugo U. R. Strand (2023)

from cpp2py.wrap_generator import *

# The module
module = module_(full_name = "pycppdlr", doc = r"", app_name = "pycppdlr")

# Imports

# Add here all includes
module.add_include("cppdlr/cppdlr.hpp")

# Add here anything to add in the C++ code at the start, e.g. namespace using
module.add_preamble("""
#include <cpp2py/converters/complex.hpp>
#include <cpp2py/converters/string.hpp>
#include <nda_py/cpp2py_converters.hpp>

using namespace cppdlr;
using namespace nda;
""")

module.add_function("""nda::vector<double> build_dlr_rf(double lambda, double eps)""")

# The class imtime_ops
c = class_(
        py_type = "ImTimeOps",  # name of the python class
        c_type = "imtime_ops",   # name of the C++ class
        doc = r"""""",   # doc of the C++ class
        hdf5 = False,
)

c.add_constructor("""(double lambda, nda::array<double, 1> dlr_rf)""", doc = r"""""")

c.add_method("""nda::array<dcomplex, 3> vals2coefs(nda::array<dcomplex, 3> g)""")
c.add_method("""nda::array<dcomplex, 3> coefs2vals(nda::array<dcomplex, 3> g)""")
c.add_method("""nda::array<dcomplex, 2> coefs2eval(nda::array<dcomplex, 3> g, double t)""")

c.add_method("""nda::vector_const_view<double> get_itnodes()""")
c.add_method("""nda::vector_const_view<double> get_rfnodes()""")
c.add_method("""nda::matrix_const_view<double> get_cf2it()""")
c.add_method("""nda::matrix_const_view<double> get_it2cf_lu()""")
c.add_method("""nda::vector_const_view<int> get_it2cf_piv()""")
c.add_method("""int rank()""")
c.add_method("""double lambda()""")

module.add_class(c)

module.generate_code()
