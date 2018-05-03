using FluxAddons

abstract type MyType end
struct MyTypeA <: MyType
  x
end
Base.size(::Type{MyTypeA}) = (6, 2, 4)

struct MyTypeB <: MyType
  x
end
Base.size(::Type{MyTypeB}) = (3, 4)

struct MyTypeC <: MyType
  x
end
Base.size(::Type{MyTypeC}) = (2, 4)


latent(x::Tuple) = map(latent, x)
latent(x) = x.latent
latent(x::MyType) = x.x
latent(x::MyType...) = map(latent, x)

batch_size = 2
function test()
  f = mlp((MyTypeA, MyTypeB), (MyTypeC, MyTypeC))
  m1, m2 = MyTypeA(rand(size(MyTypeA)..., batch_size)),
           MyTypeB(rand(size(MyTypeB)..., batch_size))
  c1, x2 = f(m1, m2)
  @test c1 isa MyTypeC
  @test c2 isa MyTypeC
end