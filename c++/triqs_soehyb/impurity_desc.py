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

c.add_constructor("""(double beta, double lambda, double eps, nda::array<dcomplex, 3> Deltat, nda::array<dcomplex, 3> F, nda::array<dcomplex, 3> F_dag, bool poledlrflag)""", doc = r"""Parameters
----------
[in]
     beta inverse temperature

[in]
     lambda DLR cutoff parameter

[in]
     eps DLR accuracy tolerance

[in]
     Deltat hybridization function in imaginary time, nda array of size r*n*n

[in]
     F impurity annihilation operator in pseudo-particle space, of size n*N*N

[in]
     F_dag impurity creation operator in pseudo-particle space, of size n*N*N

[in]
     poledlrflag flag for whether to use dlr for pole expansion. True for using dlr. False has not been implemented yet.""")

c.add_method("""nda::array<dcomplex, 3> Sigma_calc (nda::array_view<dcomplex, 3> Gt, std::string order)""",
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

c.add_method("""nda::array<dcomplex, 3> G_calc (nda::array_view<dcomplex, 3> Gt, std::string order)""",
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

module.add_class(c)



module.generate_code()