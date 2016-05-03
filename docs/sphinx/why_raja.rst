.. ##
.. ## Copyright (c) 2016, Lawrence Livermore National Security, LLC.
.. ##
.. ## Produced at the Lawrence Livermore National Laboratory.
.. ##
.. ## All rights reserved.
.. ##
.. ## For release details and restrictions, please see raja/README-license.txt
.. ##


===================================
Why RAJA?
===================================

RAJA is a system for enabling performance portable parallel codes, aiming 
to performantly run across many architectures while minimizing the burden
on developers in porting. A key difference between RAJA and other systems
is its focus on simple exposition of the  latent parallelism in existing codes,
with an extreme aversion to complicate interfaces and directives in user code.

Given the breadth of backgrounds of people who are being exposed to parllelism,
we need to ask, are you:

    * :ref:`Brand new to parallel code<new_to_parallel>`?
    * :ref:`Coming from an OpenMP/OpenACC background<from_directiveland>`?
    * :ref:`Aware of Kokkos and the other major parallelism frameworks<experienced_user>`?
    * :ref:`Looking to improve RAJA with your own fast abstractions<developers_developers_developers>`?



.. _new_to_parallel

If you're new to parallel code, you're coming to it at an interesting time.
Most old architectures were simple variants of a CPU, and code bases could
be reasonably stable while achieving reasonable performance. In the past
decade users have seen architectures involving graphics cards (NVIDIA/AMD),
Xeon Phi (Intel), and many other systems complicating this idea of 
"performance portability."

Various solutions came out. OpenMP and OpenACC, traditional parallel tools,
have been extended to target these new architectures. A system called `Kokkos <https://github.com/kokkos/kokkos>`_
came out as a way to express parallelism in your code, and allow Kokkos to
figure out how that parallelism maps to the GPU.

What sets RAJA apart is an extreme focus on ease of use while maintaining performance.
In the RAJA model, you change your for loops into RAJA parallel constructs. The bodies
of these loops often remain undisturbed. You then select a RAJA parallel execution policy
that matches the architecture on which you are running and RAJA handles the rest.
The problem of tuning stops being one of time-consuming code rewriting, and becomes one
of changing policies until you find one that works, with code perturbation being minimial.
.. ## Mention Apollo here?

.. _from_directiveland



.. _experienced_user

.. _developers_developers_developers
