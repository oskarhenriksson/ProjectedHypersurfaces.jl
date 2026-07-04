using Documenter
using ProjectedHypersurfaces

makedocs(
    sitename = "ProjectedHypersurfaces Documentation",
    modules = [ProjectedHypersurfaces],
    format = Documenter.HTML(edit_link = "main"),
    checkdocs = :exports,
)

deploydocs(
    repo = "github.com/oskarhenriksson/ProjectedHypersurfaces.jl.git",
)
