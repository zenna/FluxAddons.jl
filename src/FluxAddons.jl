module FluxAddons
using Flux
using Spec
using ZenUtils

include("mlp.jl")

Flux.children(f::Function) = ZenUtils.fields(f)
end