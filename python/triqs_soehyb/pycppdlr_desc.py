################################################################################
#
# triqs_soehyb: Sum-Of-Exponentials bold HYBridization expansion impurity solver
#
# Copyright (C) 2023 by H. U.R. Strand
#
# triqs_soehyb is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# triqs_soehyb is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# triqs_soehyb. If not, see <http://www.gnu.org/licenses/>.
#
################################################################################

# Poor man's wrapper for cppdlr::imtime_ops

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

module.add_enum(c_name = "cppdlr::statistic_t",
         c_namespace = "",
         values = ["cppdlr::Boson","cppdlr::Fermion"])

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
c.add_method("""nda::array<dcomplex, 3> reflect(nda::array<dcomplex, 3> g)""")
c.add_method("""nda::array<dcomplex, 2> coefs2eval(nda::array<dcomplex, 3> g, double t)""")

c.add_method("""nda::array<dcomplex, 3> convolve(double beta, cppdlr::statistic_t statistic, nda::array<dcomplex, 3> fc, nda::array<dcomplex, 3> gc, bool time_order = false)""")

c.add_method("""nda::vector_const_view<double> get_itnodes()""")
c.add_method("""nda::vector_const_view<double> get_rfnodes()""")
c.add_method("""nda::matrix_const_view<double> get_cf2it()""")
c.add_method("""nda::matrix_const_view<double> get_it2cf_lu()""")
c.add_method("""nda::vector_const_view<int> get_it2cf_piv()""")
c.add_method("""int rank()""")
c.add_method("""double lambda()""")

module.add_class(c)

module.generate_code()
