using Documenter, KadanoffBaym

makedocs(
    modules = [KadanoffBaym],
    sitename = "KadanoffBaym.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
    ),
    pages = [
                "Overview" => "index.md",
                "Examples" => ["examples/TightBindingModel.md",
                               "examples/FermiHubbard2B.md",
                               "examples/FermiHubbardTM.md",
                               "examples/OpenBoseDimer.md",
                               "examples/BoseEinsteinCondensate.md",
                               "examples/StochasticProcesses.md"]
            ],
    warnonly = true,
    )

deploydocs(
    repo = "github.com/NonequilibriumDynamics/KadanoffBaym.jl.git",
    devbranch = "master",
)
