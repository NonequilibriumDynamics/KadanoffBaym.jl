using Documenter, KadanoffBaym, LinearAlgebra

makedocs(
    sitename="KadanoffBaym.jl",
    pages = [
                "Overview" => "index.md",
                "Examples" => ["examples/TightBindingModel.md",
                               "examples/FermiHubbard2B.md",
                               "examples/FermiHubbardTM.md",
                               "examples/OpenBoseDimer.md",
                               "examples/StochasticProcesses.md"]
            ]
    )

deploydocs(
    repo = "github.com/NonequilibriumDynamics/KadanoffBaym.jl.git",
)
