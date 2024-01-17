# Generated automatically using the command :
# c++2py --target_file_only --cxxflags="-std=c++20" -C nda_py impurity.hpp
from cpp2py.wrap_generator import *

# The module
module = module_(full_name = "impurity", doc = r"", app_name = "impurity")

# Imports

# Add here all includes
module.add_include("impurity.hpp")

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

c.add_member(c_name = "dlr_if",
             c_type = "nda::vector<dcomplex>",
             read_only= False,
             doc = r"""""")

c.add_constructor("""(double beta, double lambda, double eps, nda::array<dcomplex, 3> F, nda::array<dcomplex, 3> F_dag)""", doc = r"""Parameters
----------
[in]
     beta inverse temperature

[in]
     lambda DLR cutoff parameter

[in]
     eps DLR accuracy tolerance

[in]
     F impurity annihilation operator in pseudo-particle space, of size n*N*N

[in]
     F_dag impurity creation operator in pseudo-particle space, of size n*N*N""")

c.add_method("""void hyb_init (nda::array<dcomplex, 3> Deltat0, bool poledlrflag = true)""",
             doc = r"""""")

c.add_method("""void hyb_decomposition (bool poledlrflag = true)""",
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

c.add_method("""nda::array<dcomplex, 3> Sigma_calc_group (nda::array<dcomplex, 3> Gt, nda::array<int, 2> D, nda::array<int, 1> diagramindex, int N)""",
             doc = r"""""")

c.add_method("""nda::array<dcomplex, 3> G_calc_group (nda::array<dcomplex, 3> Gt, nda::array<int, 2> D, nda::array<int, 1> diagramindex, int N)""",
             doc = r"""""")

c.add_method("""void copy_aaa_result (nda::vector<double> pol0, nda::array<dcomplex, 3> weights0, nda::vector<double> pol_reflect0, nda::array<dcomplex, 3> weights_reflect0)""",
             doc = r"""""")

module.add_class(c)



module.generate_code()
