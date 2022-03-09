# Parameters and architecture based on:
# https://github.com/fabio-4/JuliaRL/blob/master/algorithms/td3.jl
# https://github.com/JuliaReinforcementLearning/ReinforcementLearningZoo.jl/blob/master/src/algorithms/policy_gradient/td3.jl
# https://github.com/sfujim/TD3/blob/master/TD3.py

using CUDA
CUDA.allowscalar(false)
#----------------------------- Model Architecture -----------------------------
γ = 0.99f0     # discount rate for future rewards #Yu

τ = 5f-3       # Parameter for soft target network updates
η_act = 1f-3   # Learning rate actor 10^(-3)
η_crit = 1f-3  # Learning rate critic

#L2_DECAY = 0.01f0

init = Flux.glorot_uniform(MersenneTwister(rng_run))
w_init(dims...) = 6f-3rand(MersenneTwister(rng_run), Float32, dims...) .- 3f-3

# Optimizers
opt_crit = ADAM(η_crit)
opt_act = ADAM(η_act)

#----------------------------- Model Architecture -----------------------------
actor = Chain(
			Dense(STATE_SIZE, L1, relu, init=init),
			#LayerNorm(L1),
	      	Dense(L1, L2, relu; init=init),
			#LayerNorm(L2),
          	Dense(L2, ACTION_SIZE, tanh; init=w_init)
			) |> gpu

actor_target = deepcopy(actor) |> gpu
actor_perturb = deepcopy(actor) |> gpu

# struct Critic{C}
#     c1::C
#     c2::C
# end
# (m::Critic)(s, a) = (inp = vcat(s, a); (m.c1(inp), m.c2(inp)))
# Flux.@functor Critic

# critic = Critic(
#     Chain(
# 		Dense(STATE_SIZE + ACTION_SIZE, L1, relu, init=init) |> gpu, 
# 		Dense(L1, L2, relu, init=init) |> gpu, 
# 		Dense(L2, 1, init=w_init) |> gpu),
#     Chain(
# 		Dense(STATE_SIZE + ACTION_SIZE, L1, relu, init=init) |> gpu, 
# 		Dense(L1, L2, relu, init=init) |> gpu, 
# 		Dense(L2, 1, init=w_init) |> gpu)
# ) 
# critic_target = deepcopy(critic) |> gpu

critic1 = Chain(
			Dense(STATE_SIZE + ACTION_SIZE, L1, relu, init=init) |> gpu,
			Dense(L1, L2, relu, init=init) |> gpu,
  			Dense(L2, 1, init=w_init) |> gpu)

critic_target1 = deepcopy(critic1) |> gpu

critic2 = Chain(
			Dense(STATE_SIZE + ACTION_SIZE, L1, relu, init=init) |> gpu,
			Dense(L1, L2, relu, init=init) |> gpu,
  			Dense(L2, 1, init=w_init) |> gpu)

critic_target2 = deepcopy(critic2) |> gpu

# ------------------------------- Action Noise --------------------------------
function sample_noise(ou::OUNoise, rng_rpl) #from 1
  Random.seed!(rng_rpl)
  dx     = ou.θ .* (ou.μ .- ou.X) .* ou.dt
  dx   .+= ou.σ .* sqrt(ou.dt) .* randn(length(ou.X)) #|> gpu
  ou.X .+= dx
  return Float32.(ou.X)
end

function sample_noise(gn::GNoise, rng_rpl; target=false) # Normal distribution
  Random.seed!(rng_rpl)
  if target == false
  	d = Normal(gn.μ, gn.σ_act)
  elseif target == true
	d = Normal(gn.μ, gn.σ_trg)
  end
  return Float32.(rand(d, ACTION_SIZE))
end

function sample_noise(pn::ParamNoise) # Normal distribution
	d = Normal(pn.μ, pn.σ_current)
	return Float32(rand(d))
  end

# function sample_noise(en::EpsNoise) #from 1
#   en.ξ = Float32(max(0.5 - en.ζ * (current_episode - MEM_SIZE/EP_LENGTH["train"]), en.ξ_min))
#   return en.ξ
# end

function adapt_param_noise!(s_norm, rng_rpl)
	a = actor(s_norm)
	# add perturbation to perturb network
	add_perturb!(rng_rpl)
	a_perturb = actor_perturb(s_norm)
	distance = sqrt(Flux.mse(a, a_perturb))

	if distance > pn.σ_target
		pn.σ_current /= pn.adoption
	else
		pn.σ_current *= pn.adoption
	end
	return nothing
end

function add_perturb!(rng_rpl)
	global actor_perturb = deepcopy(actor) |> gpu

	Random.seed!(rng_rpl)
	for p_t in Flux.params(actor_perturb)
	  p_t .= p_t .+ sample_noise(pn)
	end
  end

# ---------------------- Param Update Functions --------------------------------
function soft_update!(target, model; τ = 1f0)
  for (p_t, p_m) in zip(Flux.params(target), Flux.params(model))
    p_t .= (1f0 - τ) * p_t .+ τ * p_m
  end
end

function update_model!(model, opt, loss, inp...)
  grads = gradient(() -> loss(model, inp...), Flux.params(model))
  update!(opt, Flux.params(model), grads)
end

# ---------------------------------- Training ----------------------------------
Flux.Zygote.@nograd Flux.params

function loss_crit(model, y, s_norm, a)
  #q1, q2 = model(s_norm, a)
  q1 = model(vcat(s_norm, a))
  #return 0.5 .* (Flux.mse(q1, y) + Flux.mse(q2, y)) |> gpu
  return  Flux.mse(q1, y) |> gpu
end

function loss_act(model, s_norm)
  actions = model(s_norm)  |> gpu
  #return -mean(critic(s_norm, actions)[1]) |> gpu # take only critic 1
  return -mean(critic1(vcat(s_norm, actions))) |> gpu # take only critic 1
end

function replay(;rng_rpl=0)
	# retrieve minibatch from replay buffer
	s, a, r, s′, done = getData(BATCH_SIZE, rng_dt=rng_rpl) |> gpu
  
	# update pertubable actor
	if noise_type == "pn"
		adapt_param_noise!(normalize(s), rng_rpl)
	end
  
	a′, n = act(actor_target, normalize(s′), train=true,
					noiseclamp=true, rng_act=rng_rpl) |> gpu

	# min target value against overestimation bias of the q-function
	#q′_min = min.(critic_target(normalize(s′), a′)...) |> gpu
	q′_min = min.(critic_target1(vcat(normalize(s′), a′)), critic_target2(vcat(normalize(s′), a′))) |> gpu
	#println(q′_min, size(q′_min))
	y = r .+ γ .* (1. .- done) .* q′_min |> gpu
  
	# update critic
	#update_model!(critic, opt_crit, loss_crit, y, normalize(s), a)
	update_model!(critic1, opt_crit, loss_crit, y, normalize(s), a)
	update_model!(critic2, opt_crit, loss_crit, y, normalize(s), a)
	
	# delay update
	if rng_rpl % 2 == 0
		# update actor
		update_model!(actor, opt_act, loss_act, normalize(s))
		# update target networks
		soft_update!(actor_target, actor; τ = τ)
		# soft_update!(critic_target, critic; τ = τ)
		soft_update!(critic_target1, critic1; τ = τ)
		soft_update!(critic_target2, critic2; τ = τ)
	end
	return nothing
end

# Choose action according to policy
function act(model, s_norm; train=true, noiseclamp=false, rng_act=0)
	act_pred = model(s_norm) |> cpu
	noise = zeros(Float32, size(act_pred))
	if train == true
		if noiseclamp == true # target network -> gaussian noise
			noise = reduce(hcat, [sample_noise(gn, rng_act + i, target=noiseclamp) for i in 1:size(act_pred)[2]])
			noise = clamp.(noise, -5f-1, 5f-1)
		elseif noise_type == "pn" # parameter noise
			add_perturb!(rng_act)
			act_pred = actor_perturb(s_norm) |> cpu
			noise = pn.σ_current
			return clamp.(act_pred, -1f0, 1f0), mean(noise)
		elseif noise_type == "ou" # Ornstein-Uhlenbeck noise
			noise = reduce(hcat, [sample_noise(ou, rng_act + i) for i in 1:size(act_pred)[2]])
		elseif noise_type == "gn" #Gaussian noise
			noise = reduce(hcat, [sample_noise(gn, rng_act + i, target=noiseclamp) for i in 1:size(act_pred)[2]])
		end
	# 	# #----------------- Epsilon noise ------------------------------
	# 	# eps = sample_noise(en, rng_noi=rng_act) # add noise only in training / choose noise
	# 	# rng=rand(MersenneTwister(rng_act))
	# 	# if rng > eps
	# 	# 	return act_pred, 0f0
	# 	# elseif rng <= eps
	# 	# 	noise = noisescale .* sample_noise(ou, rng_noi=rng_act)
	# 	#noise = noisescale .* randn(MersenneTwister(rng_act), Float32, size(act_pred))
		#noise = noiseclamp ? clamp.(noise, -5f-1, 5f-1) : noise #add noise clamp?
	# 	# 	# return action, eps
	# 	# end
	# 	#-------------------------------------------------------------
	return clamp.(act_pred .+ noise, -1f0, 1f0), mean(noise)
	end
	return clamp.(act_pred .+ noise, -1f0, 1f0), mean(noise)
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
  local results = Array{Float64}(undef, 0, 27)
  for step=1:NUM_STEPS
	# create individual random seed
	rng_step = parse(Int, string(rng_ep)*string(step))
	# determine action
	s = copy(env.state)
	a, noise = act(actor, normalize(s |> gpu),  noiseclamp=false,
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
      remember(s, a, r, s′, finished(env, s′))
	  # update network weights
	  replay(rng_rpl=rng_step)
	  # break episode in training
	  ##finished(env, s′) && break
    end
  end

  if track == 0
	  return (reward_eps / last_step), last_step, (noise_eps / last_step)
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
		total_reward[i], last_step, noise_mean[i] = episode!(env_train, train=true, render=render,
													track=track, rng_ep=rng_ep)
		print("Episode: $(@sprintf "%4i" i) | Mean Score: $(@sprintf "%7.2f" total_reward[i]) "*
				"| # steps: $(@sprintf "%2i" last_step) | Noise: $(@sprintf "%7.4f" noise_mean[i]) | ")
									
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
			print("Eval score $(@sprintf "%7.3f" score_mean[idx]) | ")
			#save weights for early stopping
			# if score_mean[idx] > best_score
			# 	# save network weights
			# 	saveBSON(actor,	total_reward, score_mean, best_run, noise_mean,
			# 						idx=i, path="temp", rng=rng_run)
			# 	# set new best score
			# 	best_score = score_mean[idx]
			# 	# save best run
			# 	global best_run = i
			# end
		end
		t_elap = round(now()-t_start, Dates.Minute)
		println("Time elapsed: $(t_elap)")
	end
	return nothing
end
