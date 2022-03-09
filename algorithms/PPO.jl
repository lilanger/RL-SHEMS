# Parameters and architecture based on:
# https://github.com/fabio-4/JuliaRL/blob/master/algorithms/ppocont.jl
# https://github.com/JuliaReinforcementLearning/ReinforcementLearningZoo.jl/blob/master/src/algorithms/policy_gradient/ppo.jl
using CUDA
CUDA.allowscalar(true)
#----
nsteps = EP_LENGTH["train"]
nenv = 10
nbatch = nenv * nsteps
memoryPPO = CircularBuffer{Any}(nbatch)
atrainiters = 50
ctrainiters = 80
targetkl=15f-3
ϵ=2f-1

function getDataPPO(batch_size; rng_dt=0)
  # Getting data in shape
  minibatch = sample(MersenneTwister(rng_dt), memoryPPO, batch_size)

  s       =   reduce(hcat,getindex.(minibatch,1))  |> gpu
  a       =   reduce(hcat,getindex.(minibatch,2))  |> gpu
  oldlogp =   reduce(hcat,getindex.(minibatch,3))  |> gpu
  r 	  =   reduce(hcat,getindex.(minibatch,4))  |> gpu
  v       =   reduce(hcat,getindex.(minibatch,5))  |> gpu
 # s_mask  = .!hcat(x[5, :]...) |> gpu #only used for final state

  return s, a, oldlogp, r, v
end

#---------------------------- Helper MEMORY --------------------------------
remember(state, action, loga, r, value) = 	push!(memoryPPO, [state, action, loga, r, value])

#----------------------------- Model Architecture -----------------------------
γ = 0.995f0     # discount rate for future rewards #Yu

τ = 1f-3       # Parameter for soft target network updates
η_act = 1f-4   # Learning rate actor 10^(-4)
η_crit = 1f-3  # Learning rate critic

α = 0.2f0		# temperature trade-off entropy/rewards

#L2_DECAY = 0.01f0

init = Flux.glorot_uniform(MersenneTwister(rng_run))
init_final(dims...) = 6f-3rand(MersenneTwister(rng_run), Float32, dims...) .- 3f-3

opt_crit = ADAM(η_crit)
opt_act = ADAM(η_act)

struct Actor{S, A1, A2}
    model::S
    μ::A1
    logσ::A2
end

Flux.@functor Actor
(m::Actor)(s) = (l = m.model(s); (m.μ(l), m.logσ(l)))

actor = Actor(
    		Chain(Dense(STATE_SIZE, L1, relu, init=init), Dense(L1, L2, relu, init=init)) |> gpu,
    		Chain(Dense(L2, ACTION_SIZE, tanh, init=init_final)) |> gpu,
    		Chain(Dense(L2, ACTION_SIZE, init=init_final)) |> gpu
			)

critic = Chain(
			Dense(STATE_SIZE, L1, relu, init=init) |> gpu,
			Dense(L1, L2, relu, init=init) |> gpu,
  			Dense(L2, 1, init=init_final) |> gpu
			)

# ---------------------- Param Update Functions --------------------------------
function soft_update!(target, model; τ = 1f0)
  for (p_t, p_m) in zip(params(target), params(model))
    p_t .= (1f0 - τ) * p_t .+ τ * p_m
  end
end

# ---------------------------------- Training ----------------------------------
Flux.Zygote.@nograd Flux.params

function replay(;rng_rpl=0)
  # retrieve minibatch from replay buffer
  s, a, oldlogp, r, adv = getDataPPO(BATCH_SIZE, rng_dt=rng_rpl) |> gpu
  for _ in 1:atrainiters
	  loga, back = pullback(params(actor)) do
		  μ, logσ = actor(normalize(s)) |> gpu
		  return sum(loglikelihoodPPO(a, μ, logσ), dims=1)
	  end
	  approxkl(loga, oldlogp) > targetkl && break
	  gs = gradient(logp -> ploss(logp, oldlogp, adv, ϵ), loga)[1]
	  update!(opt_act, params(actor), back(gs))
  end

  Flux.train!(params(critic), Iterators.repeated((s, r), ctrainiters), opt_crit) do s, r
	return Flux.mse(critic(normalize(s)), r)
  end
  return nothing
end

function loglikelihoodPPO(x, mu, logσ)
    return -5f-1 .* (((x .- mu) ./ (exp.(logσ) .+ 1f-8)) .^ 2f0 .+ 2f0 .* logσ .+ log(2f0*Float32(π)))
end

# Choose action according to policy
function act(rng::MersenneTwister, s_norm; train::Bool=true)
	μ, logσ = actor(s_norm) |> gpu
	σ = exp.(logσ) |> gpu
	ã = μ .+ σ .* (randn(rng, eltype(logσ), size(σ)) |> gpu)
	logpã   = sum(loglikelihoodPPO(ã, μ, logσ), dims=1)
    return ã, logpã
end

approxkl(logp, oldlogp) = mean(oldlogp .- logp)

function ploss(logp, oldlogp, adv, ϵ)
    rat = exp.(logp .- oldlogp)
    return -mean(min.(rat .* adv, min.(max.(rat, one(ϵ)-ϵ), one(ϵ)+ϵ) .* adv))
end

function scale_action(action)
	#scale action [-1, 1] to action bounds
	scaled_action = Float32.(ACTION_BOUND_LO .+
						(action .+ ones(ACTION_SIZE)) .*
							0.5 .* (ACTION_BOUND_HI .- ACTION_BOUND_LO))
	return scaled_action
end

function episode!(env::Shems; NUM_STEPS=EP_LENGTH["train"], train=true, render=false,
					track=0, rng_ep=0)
  reset!(env, rng=rng_ep) # rng = -1 sets evaluation/test initials
  local reward_eps=0f0
  local noise_eps=0f0
  local last_step = 1
  local results = Array{Float64}(undef, 0, 25)
  for step=1:NUM_STEPS
	# create individual random seed
	rng_step = parse(Int, string(abs(rng_ep))*string(step))
	# determine action
	s = copy(env.state)
	a, loga = act(MersenneTwister(rng_step), normalize(s |> gpu), train=train) |> cpu
	v = critic(normalize(s |> gpu))
	scaled_action = scale_action(a)

	# execute action in RL environment
	if track == 0
		r, s′ = step!(env, s, scaled_action)
	elseif track == 1 # DRL
		r, s′, results_new = step!(env, s, scaled_action, track=track)
		results = vcat(results, results_new)
	elseif track < 0 # rule-based
		a = action(env, track)
		r, s′, results_new = step!(env, s, a, track=track)
		results = vcat(results, results_new)
	end

	# render step
	if render == true
		#sleep(1f-10)
		gui(plot(env))
		#plot(env) # for gif creation
		#frame(anim) # for gif creation
	end
	reward_eps += r
	#noise_eps += noise
	last_step = step

	# save step in replay buffer
	if train == true
      remember(s, a, loga, r, v)  #, finished(env, s′)) # for final state
	  # update network weights
	  replay(rng_rpl=rng_step)
	  # break episode in training
	  finished(env, s′) && break
    end
  end

  if track == 0
	  return (reward_eps / last_step), (noise_eps / last_step)
  else
      return (reward_eps / last_step), results
  end
end

function run_episodes(env_train::Shems, env_eval::Shems, total_reward, score_mean, best_run,
						noise_mean, test_every, render, rng; track=0)
	best_score = -1000
	for i=1:NUM_EP
		score=0f0
		score_all=0f0
		global current_episode = i
		# set individual seed per episode
		rng_ep = parse(Int, string(rng)*string(i))

		# Train set
		total_reward[i], noise_mean[i] = episode!(env_train, train=true, render=render,
													track=track, rng_ep=rng_ep)
		print("Episode: $i | Mean Step Score: $(@sprintf "%9.3f" total_reward[i]) |  $(pn.σ_current)  |  ")

		# test on eval data set
		if i % test_every == 1
			idx = ceil(Int32, i/test_every)
			# Evaluation set
			for test_ep in 1:test_runs
				rng_test= parse(Int, string(seed_ini)*string(test_ep))
				score, noise = episode!(env_eval, train=false, render=false,
										NUM_STEPS=EP_LENGTH["train"], track=track, rng_ep=rng_test)
				score_all += score
			end
			score_mean[idx] = score_all/test_runs
			print("Eval score $(@sprintf "%9.3f" score_mean[idx]) | ")
			#save weights for early stopping
			if score_mean[idx] > best_score
				# save network weights
				saveBSON(actor,	total_reward, score_mean, best_run, noise_mean,
									idx=i, path="temp", rng=rng_run)
				# set new best score
				best_score = score_mean[idx]
				# save best run
				global best_run = i
			end
		end
		t_elap = round(now()-t_start, Dates.Minute)
		println("Time elapsed: $(t_elap)")
	end
	return nothing
end
