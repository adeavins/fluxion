****
TODO
****

Milestones
==========

This is the approximate sequence of adding features towards the first beta:

* ODE with integer dimensions
* ODE with several fields
* 0-dimensional SDE
* 1+-dimensional SDE with the noise depending on transverse dimensions
* Matrix operations over integer dimensions
* Einstein summation over integer dimensions
* Step error estimation (and, possibly, solution extrapolation based on the stepper)

There are also side tasks that are mostly independent from the main interface:

* High-order noises
* More steppers. We will need at least a good fixed-step one (e.g. RK4), a good adaptive-step one (TBD), and a good stochastic one (e.g. midpoint)
* Helper functions for fields (e.g. integration)
* A debug plotting function (essentially, plots whatever ``integrate()`` returns; very good for demonstrations)
* Ability to save separate samples for all trajectories/all ensembles during SDE integration
* Interaction with ``sympy`` (e.g. transforming equations to and from ``sympy`` expressions)
* Dynamic indication of the integration progress
* A step-by-step initialization of the integration. Something like ``integrate(eq).starting_state(...).sample(...).run()``. This will allow one to avoid giant parameter lists. Each method call returns a new immutable object, so one can possibly fork the process and create two integrators with, say, different starting states.

The third development direction is performance.
The main part of it is, of course, parallel execution in multi-core, multi-GPU and multi-node environments.
Ideally, we would want to have array operations being parallelised automatically, but this may not be easy to do.


Assorted notes
==============

It will be convenient for the user to get sampling results as fields. In order for FFT to work correctly on such fields, we must somehow know for every dimension whether its grid is uniform or not. With some degree of precision, it can be determined dynamically.

Another problem with that is finding what kind of field to save results into. Variants are:

* Deduce the field (kind and dimensions) from the first returned sample (if it is an expression). Problem: if the sampler returns different expressions at different times, our first assumption may be incorrect.
* Save a separate field for each sample, and then find the most general field to save the samples in. Safe, but leads to memory overhead (especially in the case of many small samples). Although, the overhead is only proportional to the number of samples, and it is very rarely a large number (tens of thousands max; even if it's a million, we're still looking at 10s of megabytes, which is not that bad).
* Require the field to be defined explicitly by the sampler. Safe, but leads to some redundancy.
* Hybrid variant: the user may declare the resulting field explicitly, but if he didn't, try to guess it from the samples.

I'd like to make dimension/field names optional. The only use for dimension names is to use them to create shortcuts in the field objects (e.g. to get a dimension as ``field.x``). Otherwise they are only needed if one wants to pretty print the equation, since we are comparing symbols by their canonical arguments. If the name is not set, we just create one based on ``id()`` or something.

Fields should have a pretty print option, where their dimensions, kind and data are displayed in a readable form. Perhaps close to how ``pandas`` does it.

There should be a way to specify the "reference" data in ``plot()`` (in a form of expressions or just raw data).

``find_generic_field()`` should have some way to handle new dimensions introduced by samplers, for example the mode space dimension in the soliton example. Specifically for the mode space dimensions, we can remember the "base" dimensions, and use them to determine the order. For entirely new dimensions (why would anyone actually need this?) we can just add them to the end.

``to_momentum_space`` takes too much time, because it recreates the new dimension on every call. Don't know how to fix it properly at the moment. Some kind of caching maybe?

For debug purposes it would be useful to save step sizes in the results. Perhaps, with an optional ``debug`` key on.

Provide per-observable plotting functions that will allow for more customization. ``plot()`` will just call them with default parameters for each of the observables.
