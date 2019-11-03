"""
    IVProbitShare <: GMMModel

An `IVProbitShare` model consists of outcomes, `y` ∈ (0,1),  regressors
`x` and instruments `z`.  The moment condition is

``E[ (Φ^{-1}(y) xβ)z ] = 0``

where Φ^{-1}(⋅) is the inverse of the standard normal distribution.

The dimensions of `x`, `y`, and `z` must be such that
`length(y) == size(x,1) == size(z,1)`
and
`size(x,2) ≤ size(z,2)`.
"""
struct IVProbitShare <: GMMModel
  x::Matrix{Float64}
  y::Vector{Float64}
  z::Matrix{Float64}
end

"""
    IVProbitShare(n::Integer, β::AbstractVector,
                      π::AbstractMatrix, ρ)

Simulate an IVProbitShare model.

# Arguments

- `n` number of observations
- `β` coefficients on `x`
- `π` first stage coefficients `x = z*π + v`
- `ρ` correlation between x[:,1] and structural error.

Returns an IVProbitShare GMMModel.
"""
function IVProbitShare(n::Integer, β::AbstractVector,
                      π::AbstractMatrix, ρ)
  z = randn(n, size(π)[1])
  endo = randn(n, length(β))
  x = z*π .+ endo
  ξ = rand(Normal(0,sqrt((1.0-ρ^2))),n).+endo[:,1]*ρ
  y = cdf.(Normal(), x*β .+ ξ)
  return(IVProbitShare(x,y,z))
end

number_parameters(model::IVProbitShare) = size(model.x,2)
number_observations(model::IVProbitShare) = length(model.y)
number_moments(model::IVProbitShare) = size(model.z,2)

function get_gi(model::IVProbitShare)
  function(β)
    ξ = quantile.(Normal(), model.y) .- model.x*β
    ξ.*model.z
  end
end

function gel_jump_problem(model::IVProbitShare)
  n = number_observations(model)
  d = number_parameters(model)
  k = number_moments(model)
  Ty = quantile.(Normal(),model.y)
  m = Model()
  @variable(m, 0.0 <= p[1:n] <= 1.0)
  @variable(m, θ[1:d])
  @constraint(m, prob,sum(p)==1.0)
  @constraint(m, momentcon[i=1:k], dot((Ty - model.x*θ).*model.z[:,i],p)==0.0)
  @NLobjective(m,Max, sum(log(p[i]) for i in 1:n))
  return(m)
end
