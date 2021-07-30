# Parameters and architecture based on:
# 1 https://github.com/msinto93/DDPG/blob/master/train.py
# 2 https://github.com/JuliaReinforcementLearning/ReinforcementLearningZoo.jl/blob/master/src/experiments/rl_envs/JuliaRL_DDPG_Pendulum.jl
# 3 https://github.com/FluxML/model-zoo/blob/master/contrib/games/differentiable-programming/pendulum/DDPG.jl
# 4 https://github.com/fabio-4/JuliaRL/blob/master/algorithms/ddpg.jl


#----------------------------- Model Architecture -----------------------------
γ = 0.995f0     # discount rate for future rewards #Yu

τ = 1f-3       # Parameter for soft target network updates
η_act = 1f-4   # Learning rate actor 10^(-4)
η_crit = 1f-3  # Learning rate critic

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

#----------------------------- Model Architecture -----------------------------
actor = Chain(
			Dense(STATE_SIZE, L1, relu, init=init),
			LayerNorm(L1),
	      	Dense(L1, L2, relu; init=init),
			LayerNorm(L2),
          	Dense(L2, ACTION_SIZE, tanh; init=init_final)) |> gpu

actor_target = deepcopy(actor) |> gpu
actor_perturb = deepcopy(actor) |> gpu

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

critic = crit(Chain(Dense(STATE_SIZE, L1, relu, init=init),Dense(L1, L2, init=init)) |> gpu,
  				Dense(ACTION_SIZE, L2, init=init) |> gpu,
  				Dense(L2, 1, init=init_final) |> gpu)

critic_target = deepcopy(critic) |> gpu

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

function add_perturb!(actor_perturb, rng_rpl)
  for p_t in params(actor_perturb)
    p_t .= p_t .+ sample_noise(pn, rng_noi=rng_rpl)
  end
end

function update_model!(model, opt, loss, inp...)
  grads = gradient(() -> loss(model, inp...), params(model))
  update!(opt, params(model), grads)
end

# ---------------------------------- Training ----------------------------------
Flux.Zygote.@nograd Flux.params
#L2_loss(model) = L2_DECAY * sum(x->sum(x.^2), params(model));

function loss_crit(model, y, s_norm, a)
  return Flux.mse(critic(s_norm, a), y) |> gpu
end

function loss_act(model, s_norm)
  actions = actor(s_norm)  |> gpu
  return -mean(critic(s_norm, actions)) |> gpu # sum better than mean?
end

function replay(;rng_rpl=0)
  # retrieve minibatch from replay buffer
  s, a, r, s′ = getData(BATCH_SIZE, rng_dt=rng_rpl) |> gpu # s_mask when with terminal state

  # update pertubable actor
  adapt_param_noise(actor, s, rng_rpl)

  # update networks
  a′ = actor_target(normalize(s′ |> gpu))
  v′ = critic_target(normalize(s′ |> gpu), a′) |> gpu
  y = r .+ (γ .* v′) |> gpu #no terminal reward switch off

  # update critic
  update_model!(critic, opt_crit, loss_crit, y, normalize(s |> gpu), a)
  # update actor
  update_model!(actor, opt_act, loss_act, normalize(s |> gpu))
  # update target networks
  soft_update!(actor_target, actor; τ = τ)
  soft_update!(critic_target, critic; τ = τ)
  return nothing
end

function adapt_param_noise(actor, s, rng_rpl)
	a = actor(normalize(s |> gpu))

	soft_update!(actor_perturb, actor, τ = 1f0)
	add_perturb!(actor_perturb, rng_rpl)
	a_perturb = actor_perturb(normalize(s |> gpu))
	distance = Flux.mse(a, a_perturb)

	if distance > pn.σ_target
		pn.σ_current /= pn.adoption
	else
		pn.σ_current *= pn.adoption
	end

	soft_update!(actor_perturb, actor, τ = 1f0)
	add_perturb!(actor_perturb, rng_rpl)
	return nothing
end


# Choose action according to policy
function act(s_norm; train=true, noiseclamp=false, rng_act=0)
	act_pred = actor(s_norm |> gpu) |> cpu
	noise = zeros(Float32, size(act_pred))
	if train == true
		##-------------Adaptive parameter noise -----------------------------
		act_pred = actor_perturb(s_norm |> gpu) |> cpu

		##-------------OU noise ---------------------------------------------
		# noise = reduce(hcat, [sample_noise(ou, rng_noi=rng_act) for i in 1:size(act_pred)[2]]) # add noise only in training / choose noise

		##-------------Gaussian noise ---------------------------------------
		# noise = reduce(hcat, [sample_noise(gn, rng_noi=rng_act) for i in 1:size(act_pred)[2]]) # add noise only in training / choose noise

		# #----------------- Epsilon noise ------------------------------
		# eps = sample_noise(en, rng_noi=rng_act) # add noise only in training / choose noise
		# rng=rand(MersenneTwister(rng_act))
		# if rng > eps
		# 	return act_pred, 0f0
		# elseif rng <= eps
		# 	noise = noisescale .* sample_noise(ou, rng_noi=rng_act)
		#noise = noisescale .* randn(MersenneTwister(rng_act), Float32, size(act_pred))
		# noise = noiseclamp ? clamp.(noise, -5f-1, 5f-1) : noise #add noise clamp?
		# 	# return action, eps
		# end
		#-------------------------------------------------------------
	end
	return clamp.(act_pred + noise, -1f0, 1f0), mean(noise)
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
  reset!(env, rng=rng_ep)
  local reward_eps=0f0
  local noise_eps=0f0
  local last_step = 1
  local results = Array{Float64}(undef, 0, 25)
  for step=1:NUM_STEPS
	# create individual random seed
	rng_step = parse(Int, string(rng_ep)*string(step))
	# determine action
	s = copy(env.state)
	a, noise = act(normalize(s |> gpu), noiseclamp=false,
									train=train, rng_act=rng_step) |> cpu
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
	noise_eps += noise
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
