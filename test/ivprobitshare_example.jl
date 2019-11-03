using GMMInference, Optim, LinearAlgebra, Random


function ivprobitshare_example()
    # Simulation parameters
    n = 100
    β = vec([0.2 1.])
    k = length(β)
    iv = 3
    π = vcat(0.1*I,ones(iv-k,k))
    ρ = 0.3

    Random.seed!(1234)
    logit = IVLogitShare(n,β,π,ρ)
    Random.seed!(1234)
    probit = IVProbitShare(n,β,π,ρ)

    # Run estimation for both models
    minimizers = repeat(hcat(typeof(probit), zeros(1,k)),2,1)
    minimizers[1,:]
    for (i,model) in enumerate([logit probit])
        println(typeof(model))

        # Test some functions
        @show number_parameters(model) == k
        @show number_observations(model) == n
        @show number_moments(model) == iv

        # Solve
        gi = get_gi(model)
        gmmObj = gmm_objective(model)
        # return gmmObj
        opt1 = optimize(θ->gmmObj(θ),
                    ones(number_parameters(model)), Newton(), autodiff =:forward)
        @show opt1.minimizer
        minimizers[i,:] = hcat(typeof(model), opt1.minimizer')
        println("")
    end

    @show minimizers
    return (minimizers[2,2:end] ≈ minimizers[2,2:end]) == true
end
