
# ProjectedHypersurfaces.jl

This repository implements "numerical elimination" techniques for representing, and computing the complement of, real hypersurfaces that arise through projection of a known variety.

It is based on the paper [Elimination Without Eliminating: Computing Complements of Real Hypersurfaces Using Pseudo-Witness Sets](https://arxiv.org/abs/2601.04383) by Paul Breiding, John Cobb, Aviva Englander, Nayda Farnsworth, Jon Hauenstein, Oskar Henriksson, David Johnson, Jordy Lopez Garcia, and Deepak Mundayur.

## Projected hypersurfaces

```@autodocs
Modules = [ProjectedHypersurfaces]
Pages = ["projected_hypersurfaces.jl"]
Order = [:type, :function]
```

## Pseudo-witness sets

```@autodocs
Modules = [ProjectedHypersurfaces]
Pages = ["pseduo_witness_sets.jl"]
Order = [:type, :function]
```

## Interpolation

```@autodocs
Modules = [ProjectedHypersurfaces]
Pages = ["pseduo_witness_sets.jl"]
Order = [:type, :function]
```

## Routing functions and gradient roadmats

```@autodocs
Modules = [ProjectedHypersurfaces]
Pages = ["routing_functions.jl", "critical_points.jl", "graph.jl"]
Order = [:type, :function]
```
