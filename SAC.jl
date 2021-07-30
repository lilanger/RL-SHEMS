# Parameters and architecture based on:
# https://github.com/fabio-4/JuliaRL/blob/master/algorithms/sac.jl
# https://github.com/JuliaReinforcementLearning/ReinforcementLearningZoo.jl/blob/master/src/algorithms/policy_gradient/sac.jl

#----------------------------- Model Architecture -----------------------------
γ = 0.995f0     # discount rate for future rewards #Yu

τ = 1f-3       # Parameter for soft target network updates
η_act = 1f-4   # Learning rate actor 10^(-4)
η_crit = 1f-3  # Learning rate critic

α = 0.2f0		# temperature trade-off entropy/rewards

#L2_DECAY = 0.01f0

init = Flux.glorot_uniform(MersenneTwister(rng_run))
init_final(dims...) = 6f-3rand(MersenneTwister(rng_run), Float32, dims...) .- 3f-3

# Optimizers
# with L2 regularization
#opt_crit = Flux.Optimiser(WeightDecay(L2_DECAY), ADAM(η_crit))
#opt_act = Flux.Optimiser(WeightDecay(L2_DECAY), ADAM(η_act))
# without L2 regularization
opt_crit = ADAM(η_crit)
opt_act = ADAM(η_act)


struct Actor{S, A1, A2}
    model::S
    μ::A1
    logσ::A2
end

(m::Actor)(s) = (l = m.model(s); (m.μ(l), m.logσ(l))) |> gpu
Flux.@functor Actor

actor = Actor(
    Chain(Dense(STATE_SIZE, L1, relu), Dense(L1, L2, relu)) |> gpu,
    Chain(Dense(L2, ACTION_SIZE)) |> gpu,
    Chain(Dense(L2, ACTION_SIZE, x -> min(max(x, typeof(x)(-20f0)), typeof(x)(2f0)))) |> gpu
)

# Critic model
struct crit
  state_crit
  act_crit
  sa_crit
end

Flux.@functor crit

function (c::crit)(state, action)
  s = c.state_crit(state)
  a = c.act_crit(action)
  c.sa_crit(relu.(s .+ a))
end

Base.deepcopy(c::crit) = crit(deepcopy(c.state_crit),
							  deepcopy(c.act_crit),
							  deepcopy(c.sa_crit))

critic1 = crit(Chain(Dense(STATE_SIZE, L1, relu, init=init),Dense(L1, L2, init=init)) |> gpu,
  				Dense(ACTION_SIZE, L2, init=init) |> gpu,
  				Dense(L2, 1, init=init_final) |> gpu)

critic2 = deepcopy(critic1) |> gpu
critic_target1 = deepcopy(critic1) |> gpu
critic_target2 = deepcopy(critic1) |> gpu

# ------------------------------- Action Noise --------------------------------
function sample_noise(ou::OUNoise; rng_noi=0) #from 1
  dx     = ou.θ .* (ou.μ .- ou.X) .* ou.dt
  dx   .+= ou.σ .* sqrt(ou.dt) .* randn(MersenneTwister(rng_noi), length(ou.X)) #|> gpu
  ou.X .+= dx
  return Float32.(ou.X)
end

function sample_noise(gn::GNoise; rng_noi=0) # Normal distribution
  d = Normal(gn.μ, gn.σ)
  dx = rand(MersenneTwister(rng_noi), d, ACTION_SIZE) #|>gpu
  return Float32.(dx)
end

function sample_noise(pn::ParamNoise; rng_noi=0) # Normal distribution
  d = Normal(pn.μ, pn.σ_current)
  dx = rand(MersenneTwister(rng_noi), d) #|>gpu
  return Float32(dx)
end

function sample_noise(en::EpsNoise; rng_noi=0) #from 1
  en.ξ = Float32(max(0.5 - en.ζ * (current_episode - MEM_SIZE/EP_LENGTH["train"]), en.ξ_min))
  return en.ξ
end

# ---------------------- Param Update Functions --------------------------------
function soft_update!(target, model; τ = 1f0)
  for (p_t, p_m) in zip(params(target), params(model))
    p_t .= (1f0 - τ) * p_t .+ τ * p_m
  end
end

function update_model!(model, opt, loss, inp...)
  grads = gradient(() -> loss(model, inp...), params(model))
  update!(opt, params(model), grads)
end

# ---------------------------------- Training ----------------------------------
Flux.Zygote.@nograd Flux.params

function loss_crit(model, y, s_norm, a)
  q = model(s_norm, a)
  return Flux.mse(q, y) |> gpu
end

function loss_act(model, s_norm)
  actions, log_π = act(s_norm) |> gpu
  Q_min = min.(critic1(s_norm, actions), critic2(s_norm, actions))
  return -mean(Q_min .- α .* log_π) |> gpu
end

function replay(;rng_rpl=0)
  # retrieve minibatch from replay buffer
  s, a, r, s′ = getData(BATCH_SIZE, rng_dt=rng_rpl) |> gpu
  a′, log_π  = act(normalize(s′), rng_act=rng_rpl) |> gpu

  # update networks
  v′_min = min.(critic_target1(normalize(s′), a′), critic_target2(normalize(s′), a′)) .- α .* log_π
  y = r .+ (γ .* v′_min)

  # update critic
  update_model!(critic1, opt_crit, loss_crit, y, normalize(s), a)
  update_model!(critic2, opt_crit, loss_crit, y, normalize(s), a)
  # update actor
  update_model!(actor, opt_act, loss_act, normalize(s))
  # update target networks
  soft_update!(critic_target1, critic1; τ = τ)
  soft_update!(critic_target2, critic2; τ = τ)
  return nothing
end

# Choose action according to policy
function act(s_norm; rng_act=0)
    μ, logσ = actor(s_norm)
	π_dist = Normal.(μ, exp.(logσ))
	z = rand.(MersenneTwister(rng_act), π_dist)
    logp_π  = sum(logpdf.(π_dist, z), dims = 1)
    logp_π  -= sum((2.0f0 .* (log(2.0f0) .- z - softplus.(-2.0f0 * z))), dims = 1)
    return tanh.(z), logp_π
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
	a = act(normalize(s |> gpu), rng_act=rng_step)[1] |> cpu
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
      remember(s, a, r, s′)  #, finished(env, s′)) # for final state
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
