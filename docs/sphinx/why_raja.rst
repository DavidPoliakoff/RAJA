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

Given the breadth of backgrounds of people who are being exposed to parallelism,
we need to ask, are you:

    * :ref:`newtoparallel`
    * :ref:`fromdirectiveland`
    * :ref:`experienceduser`
    * :ref:`developersdevelopersdevelopers`



.. _newtoparallel:

======================================
New to Parallel Computing?
======================================

If you're new to parallel code, you're coming to it at an interesting time.
Most old architectures were simple variants of a CPU, and code bases could
be reasonably stable while achieving reasonable performance. In the past
decade users have seen architectures involving graphics cards (NVIDIA/AMD),
Xeon Phi (Intel), and many other systems complicating this idea of 
"performance portability."

Various solutions came out. OpenMP and OpenACC, traditional parallel tools,
have been extended to target these new architectures. A system called `Kokkos <https://github.com/kokkos/kokkos>`_
came out as a way to express parallelism in your code, and allow `Kokkos <https://github.com/kokkos/kokkos>`_ to
figure out how that parallelism maps to the GPU.

What sets RAJA apart is an extreme focus on ease of use while maintaining performance.
In the RAJA model, you change your for loops into RAJA parallel constructs. The bodies
of these loops often remain undisturbed. You then select a RAJA parallel execution policy
that matches the architecture on which you are running and RAJA handles the rest.
The problem of tuning stops being one of time-consuming code rewriting, and becomes one
of changing policies until you find one that works, with code perturbation being minimial.

.. _fromdirectiveland:

===========================================================================
Coming from an OpenMP/OpenACC background?
===========================================================================
If you're an OpenMP or OpenACC developer, RAJA shouldn't be a huge shift. You're used to
marking up a parallel region, and tuning by changing the directive in a way that tells the
compiler to parallelize in a certain way. Where in a directive based approach you might find
yourself tuning a directive in many different places, in RAJA you start by finding loops which
execute in similar ways and giving them a different "execution policy" which you tune to optimize
performance. Given a code already implemented in OpenMP4 it might not be worth translating to RAJA,
but if you're making a move from OpenMP3 RAJA is a good place to start.

.. _experienceduser:

==========================================================================
Aware of Kokkos and the other major parallelism frameworks?
==========================================================================

If you currently have code running in `Kokkos <https://github.com/kokkos/kokkos>`_, the performance of RAJA and
Kokkos are close enough that we don't recommend moving to RAJA. If you're
looking at Kokkos and RAJA and trying to pick between them, there are a few
forces that push you in each direction. First, if you have an existing code,
RAJA is much more centered on exposing the latent parallelism in your code,
where Kokkos is a language extension that allows you to express a breadth
of parallelism. In the case of porting existing code, especially large codes,
the incremental approach and minimal code perturbation in RAJA are often attractive.

On the other hand, RAJA only manages how loops execute. While there is work towards
making an ecosystem around RAJA that manages data, Kokkos manages data layout
right now. If you're building a new code that is extremely sensitive to data
layout, and you don't want to manage laying out the data yourself, Kokkos
may be advantageous. Further, if your parallel constructs are not loops
(tasks, for example), Kokkos supports these currently in a way RAJA does not.

Ultimately, RAJA is about exposing latent parallelism in codes, and Kokkos is
about providing you with language to express parallelism. As code size increases
RAJA looks better, as the parallelism model moves farther from parallel loops Kokkos
looks better.

.. _developersdevelopersdevelopers:

==========================================================================
Looking to improve RAJA with your own fast abstractions?
==========================================================================

Extension section here
