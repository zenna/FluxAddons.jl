pass(x) = (print(size(x)); x)
pass(xs...) = (print(map(size, xs)); xs)
pass(xs::Tuple) = (print(map(size, xs)); xs)

"NTuple of where N unspecified"
UTuple{T} = Tuple{Vararg{T, N}} where N

"Size an array"
Size{N} = NTuple{N, Int}
nelems(sz::Size) = prod(sz)
nelems(sz::UTuple{Size}) = sum(prod.(sz))

"Multilayer perceptron of `nin` inputs, `nout` outputs"
function mlp(nin::Integer, nout::Integer; nmids=[], σ=Flux.elu)
  ls = [nin; nmids...; nout]
  layers = [Dense(ls[i], ls[i+1], σ) for i = 1:length(ls)-1]
  Chain(layers...)
end

"Number of elements per batch (assiming batch_dim is last)"
nelemperbatch(x) = prod(size(x)[1:end-1])

unroll(x::AbstractArray) = reshape(x, (nelemperbatch(x), size(x)[end]))
unroll(xs::AbstractArray...) = map(unroll, xs)
unroll(xs::UTuple{AbstractArray}) = map(unroll, xs)

function invcat(xs::AbstractArray, outszs::UTuple{Size})
  @pre sum(prod.(outszs)) == size(xs, 1)
  ZenUtils.invcat(xs, prod.(outszs), 1)
end

splatvcat(x) = x
splatvcat(xs...) = vcat(xs...)
splatvcat(xs::Tuple) = vcat(xs...)

batchreshape(x, shape) = reshape(x, (shape...,size(x)[end]))

function postprocess(outszs::UTuple{Size})
  (xs -> map(batchreshape, xs, outszs)) ∘ (xs -> invcat(xs, outszs))
end

"Mapping from `Array`s to reshaped arrays"
postprocess(outsz::Size) = xs -> batchreshape(xs, outsz)

"Mulilayer perceptron"
function mlp(inszs::Union{UTuple{Size}, Size}, outszs::Union{UTuple{Size}, Size}; kwargs...)
  preprocess = splatvcat ∘ unroll
  net = mlp(nelems(inszs), nelems(outszs); kwargs...)
  postprocess(outszs) ∘ net ∘ preprocess
end

data(x) = x.data
data(xs...) = map(data, xs)

Base.size(szs::UTuple{Type}) = size.(szs)

repackage(outtype::Type) = xs -> outtype(xs)

"Apply T1(x1), T2(x2), ..."
repackage(outtypes::UTuple{Type}) = xs -> map(|>, xs, outtypes)

function mlp(intypes::Union{UTuple{Type}, Type}, outtypes::Union{UTuple{Type}, Type}; kwargs...)
  net = mlp(size(intypes), size(outtypes); kwargs...) 
  repackage(outtypes) ∘ net ∘ data
end