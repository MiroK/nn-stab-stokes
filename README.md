Installation
------------

The scripts can be executed within the pyadjoint docker image (http://www.dolfin-adjoint.org/en/latest/download/index.html)

```
fenicsproject run quay.io/dolfinadjoint/pyadjoint:master
```

# TODO
Some ideas related to the project can be found in the [notes](https://github.com/MiroK/nn-stab-stokes/blob/master/doc/notes.pdf).
## Methodology
 - [ ] What does the learned operator look like/properties?
 - [ ] Say we learn on one-few resolutions and fixed example (think bcs in Stokes); does that stab help in other resulutions/examples?
 - [ ] Are we learning some stabilization?
 - [ ] PDE cstr literature on problems we're looking at - one shot?

## Infrastructure
 - [ ] Keras like spec of the NN model with possibility to set(init) weights, store/load model
 - [ ] Stability testing
