using Documenter
using ProjectedHypersurfaceRegions

makedocs(
    sitename = "ProjectedHypersurfaceRegions Documentation",
    modules = [ProjectedHypersurfaceRegions],
    format = Documenter.HTML(edit_link = "main"),
    checkdocs = :exports,
)

deploydocs(
    repo = "github.com/oskarhenriksson/ProjectedHypersurfaceRegions.jl.git",
)
