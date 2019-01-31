module FluxAddons
using Flux
using Spec
# using ZenUtilsScene

include("mlp.jl")

export mlp

# Flux.children(f::Function) = ZenUtils.fields(f)
end