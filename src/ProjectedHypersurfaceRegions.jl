module ProjectedHypersurfaceRegions
using HomotopyContinuation, LinearAlgebra, OrdinaryDiffEq, SciMLBase, LightGraphs, ProgressMeter
using LRUCache: LRU

const HC = HomotopyContinuation
const DE = OrdinaryDiffEq

import HomotopyContinuation.evaluate!
import HomotopyContinuation.evaluate_and_jacobian!
import HomotopyContinuation.evaluate
import HomotopyContinuation.taylor!
import HomotopyContinuation.ModelKit.evaluate
import HomotopyContinuation.ModelKit.nvariables
import HomotopyContinuation.ModelKit.variables

using Reexport: @reexport
@reexport using HomotopyContinuation

include("restriction_to_line_system.jl")
include("pseudo_witness_sets.jl")
include("gradient_cache.jl")
include("hypersurfaces.jl")
include("routing_functions.jl")
include("homotopy.jl")
include("critical_points.jl")
include("ode_solving.jl")
include("graph.jl")
include("plotting.jl")



end
