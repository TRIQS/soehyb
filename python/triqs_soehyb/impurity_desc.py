################################################################################
#
# triqs_soehyb - Sum-Of-Exponentials bold HYBridization expansion impurity solver
#
# Copyright (C) 2025 by H. U.R. Strand
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
# TRIQS. If not, see <http://www.gnu.org/licenses/>.
#
################################################################################

# Manually wrapped methods

from cpp2py.wrap_generator import *

# The module
module = module_(full_name = "impurity", doc = r"", app_name = "impurity")

# Imports

# Add here all includes
module.add_include("triqs_soehyb/impurity.hpp")
module.add_include("triqs_soehyb/dlr_dyson_ppsc.hpp")

# Add here anything to add in the C++ code at the start, e.g. namespace using
module.add_preamble("""
#include <cpp2py/converters/complex.hpp>
#include <cpp2py/converters/string.hpp>
#include <nda_py/cpp2py_converters.hpp>

""")


# The class fastdiagram
c = class_(
        py_type = "Fastdiagram",  # name of the python class
        c_type = "fastdiagram",   # name of the C++ class
        doc = r"""""",   # doc of the C++ class
        hdf5 = False,
)

c.add_member(c_name = "Deltaiw",
             c_type = "nda::array<dcomplex, 3>",
             read_only= False,
             doc = r"""""")

c.add_member(c_name = "Deltaiw_reflect",
             c_type = "nda::array<dcomplex, 3>",
             read_only= False,
             doc = r"""""")

c.add_member(c_name = "dlr_if_dense",
             c_type = "nda::vector<dcomplex>",
             read_only= False,
             doc = r"""""")

c.add_constructor("""(double beta, double lambda, imtime_ops itops, nda::array<dcomplex, 3> F, nda::array<dcomplex, 3> F_dag)""", doc = r"""Parameters
----------
[in]
     beta inverse temperature

[in]
     lambda DLR cutoff parameter

[in]
     itops DLR imaginary time operator class instance

[in]
     F impurity annihilation operator in pseudo-particle space, of size n*N*N

[in]
     F_dag impurity creation operator in pseudo-particle space, of size n*N*N""")

c.add_method("""void hyb_init (nda::array<dcomplex, 3> Deltat0, bool poledlrflag = true)""",
             doc = r"""""")

c.add_method("""void hyb_decomposition (bool poledlrflag = true, double eps = 0.0)""",
             doc = r"""Parameters
----------
[in]
     Deltat hybridization function in imaginary time, nda array of size r*n*n

[in]
     poledlrflag flag for whether to use dlr for pole expansion. True for using dlr. False has not been implemented yet.""")

c.add_method("""nda::vector<dcomplex> get_it_actual ()""",
             doc = r"""""")

c.add_method("""nda::array<dcomplex, 3> free_greens (double beta, nda::array<dcomplex, 2> H_S, double mu = 0.0, bool time_order = false)""",
             doc = r"""""")

c.add_method("""nda::array<dcomplex, 3> free_greens_ppsc (double beta, nda::array<dcomplex, 2> H_S)""",
             doc = r"""""")

c.add_method("""double partition_function (nda::array<dcomplex, 3> Gt)""",
             doc = r"""""")

c.add_method("""nda::array<dcomplex, 3> Sigma_calc (nda::array<dcomplex, 3> Gt, std::string order)""",
             doc = r"""Parameters
----------
[in]
     Gt pseudo-particle Green's function G(t), of size r*N*N

[in]
     order diagram order: "NCA", "OCA" or "TCA"

Returns
-------
out
     pseudo-particle self energy diagram, r*N*N""")

c.add_method("""nda::array<dcomplex, 3> G_calc (nda::array<dcomplex, 3> Gt, std::string order)""",
             doc = r"""Parameters
----------
[in]
     Gt pseudo-particle Green's function G(t), of size r*N*N

[in]
     order diagram order: "NCA", "OCA" or "TCA"

Returns
-------
out
     impurity Green's function diagram, r*n*n""")

c.add_method("""nda::array<dcomplex, 3> time_ordered_dyson (double beta, nda::array<dcomplex, 2> H_S, double eta_0, nda::array_view<dcomplex, 3> Sigma_t)""",
             doc = r"""""")

c.add_method("""int number_of_diagrams (int m)""",
             doc = r"""""")

c.add_method("""nda::array<dcomplex, 3> Sigma_calc_group (nda::array<dcomplex, 3> Gt, nda::array<int, 2> D, nda::array<int, 1> diagramindex)""",
             doc = r"""""")

c.add_method("""nda::array<dcomplex, 3> G_calc_group (nda::array<dcomplex, 3> Gt, nda::array<int, 2> D, nda::array<int, 1> diagramindex)""",
             doc = r"""""")

c.add_method("""void copy_aaa_result (nda::vector<double> pol0, nda::array<dcomplex, 3> weights0)""",
             doc = r"""""")

module.add_class(c)


# The class dyson_it_ppsc
c = class_(
        py_type = "DysonItPPSC",  # name of the python class
        c_type = "dyson_it_ppsc<nda::array<dcomplex, 2> >",   # name of the C++ class
        doc = r"""""",   # doc of the C++ class
        hdf5 = False,
)

c.add_constructor("""(double beta, imtime_ops itops, nda::array<dcomplex, 2> H)""", doc = r"""""")

c.add_method("""nda::array<dcomplex, 3> solve(nda::array_view<dcomplex, 3> Sigma_t, double eta)""",
             doc = r"""""")

c.add_method("""nda::array<dcomplex, 3> solve_with_op(nda::array_view<dcomplex, 3> Sigma_t, double eta, nda::array_view<dcomplex, 2> op)""",
             doc = r"""""")

module.add_class(c)


module.generate_code()
