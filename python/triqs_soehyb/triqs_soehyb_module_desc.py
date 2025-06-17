# Generated automatically using the command :
# c++2py ../../c++/triqs_soehyb/triqs_soehyb.hpp -p --members_read_only -N triqs_soehyb -a triqs_soehyb -m triqs_soehyb_module -o triqs_soehyb_module --moduledoc="The triqs_soehyb python module" -C triqs --cxxflags="-std=c++17" --target_file_only
from cpp2py.wrap_generator import *

# The module
module = module_(full_name = "triqs_soehyb_module", doc = r"The triqs_soehyb python module", app_name = "triqs_soehyb")

# Imports

# Add here all includes
module.add_include("triqs_soehyb/triqs_soehyb.hpp")

# Add here anything to add in the C++ code at the start, e.g. namespace using
module.add_preamble("""
#include <cpp2py/converters/string.hpp>

using namespace triqs_soehyb;
""")


# The class toto
c = class_(
        py_type = "Toto",  # name of the python class
        c_type = "triqs_soehyb::toto",   # name of the C++ class
        doc = r"""A very useful and important class""",   # doc of the C++ class
        hdf5 = True,
        arithmetic = ("add_only"),
        comparisons = "==",
        serializable = "tuple"
)

c.add_constructor("""()""", doc = r"""""")

c.add_constructor("""(int i_)""", doc = r"""Construct from integer

Parameters
----------
i_
     a scalar  :math:`G(\tau)`""")

c.add_method("""int f (int u)""",
             doc = r"""A simple function with :math:`G(\tau)`

Parameters
----------
u
     Nothing useful""")

c.add_method("""std::string hdf5_format ()""",
             is_static = True,
             doc = r"""HDF5""")

c.add_property(name = "i",
               getter = cfunction("int get_i ()"),
               doc = r"""Simple accessor""")

module.add_class(c)

module.add_function ("int triqs_soehyb::chain (int i, int j)", doc = r"""Chain digits of two integers

Parameters
----------
i
     The first integer

j
     The second integer

Returns
-------
out
     An integer containing the digits of both i and j""")



module.generate_code()
