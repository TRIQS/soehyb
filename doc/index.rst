.. _welcome:

The TRIQS Sum-Of-Exponentials bold HYBridization expansion impurity solver (triqs_soehyb)
*********

.. sidebar:: triqs_soehyb |PROJECT_VERSION|

   This is the homepage of triqs_soehyb |PROJECT_VERSION|.
   For changes see the :ref:`changelog page <changelog>`.
      
      .. image:: _static/logo_github.png
         :width: 75%
         :align: center
         :target: https://github.com/triqs/soehyb


The TRIQS Sum-Of-Exponentials bold HYBridization expansion impurity solver (triqs_soehyb) can solve generic multiband Anderson impurity problems approximately by perturbative expansion in the coupling to the environment (a.k.a. the hybridization function :math:`\Delta(\tau)`).

This implementation is leveraging the sum-of-exponentials (SOE) trick to acheive parametrically better computational complexity compared to direct integration and uses the Discrete Lehmann Representation (DLR) for representing response functions.

Learn how to use triqs_soehyb in the :ref:`documentation`.

    
.. toctree::
   :maxdepth: 2
   :hidden:

   install
   documentation
   issues
   ChangeLog.md
   about
