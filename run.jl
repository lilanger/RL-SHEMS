# Parameters and architecture based on:
# 1 https://github.com/msinto93/DDPG/blob/master/train.py
# 2 https://github.com/JuliaReinforcementLearning/ReinforcementLearningZoo.jl/blob/master/src/experiments/rl_envs/JuliaRL_DDPG_Pendulum.jl
# 3 https://github.com/FluxML/model-zoo/blob/master/contrib/games/differentiable-programming/pendulum/DDPG.jl
# 4 https://github.com/fabio-4/JuliaRL/blob/master/algorithms/ddpg.jl

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

function sample_noise(en::EpsNoise; rng_noi=0) #from 1
  en.ξ = Float32(max(0.5 - en.ζ * (current_episode - MEM_SIZE/EP_LENGTH["train"]), en.ξ_min))
  return en.ξ
end

# ---------------------- Param Update Functions --------------------------------
function update_target!(target, model; τ = 1f0)
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
#L2_loss(model) = L2_DECAY * sum(x->sum(x.^2), params(model));

loss_crit(model, y, s, a) = Flux.mse(critic(s, a), y); #+ L2_loss(model)

function loss_act(model, s_norm, rng_act)
  actions = act(model, s_norm, rng_act=rng_act)
  return -mean(critic(s_norm, actions))  # sum better than mean?
end

function replay(;rng_rpl=0)
  # retrieve minibatch from replay buffer
  s, a, r, s′ = getData(BATCH_SIZE, rng_dt=rng_rpl) # s_mask when with terminal state

  a′ = act(actor_target, normalize(s′ |> gpu), noisescale= noisescale_trg, noiseclamp=true, rng_act=rng_rpl)
  v′ = critic_target(normalize(s′ |> gpu), a′)
  y = r .+ (γ * v′) #no terminal reward switch off

  # update critic, actor
  update_model!(critic, opt_crit, loss_crit, y, normalize(s |> gpu), a)
  update_model!(actor, opt_act, loss_act, normalize(s |> gpu), rng_rpl)

  # update target networks
  update_target!(actor_target, actor; τ = τ)
  update_target!(critic_target, critic; τ = τ)
  return nothing
end

# Choose action according to policy
function act(actor, s; noisescale=noisescale, train=true, noiseclamp=false, rng_act=0)
	act_pred = actor(normalize(s |> gpu) |> gpu) |> cpu
	noise = zeros(ACTION_SIZE)
	if train == true
		# noise = noisescale .* sample_noise(ou, rng_noi=rng_act)   # add noise only in training / choose noise
		# noise = noisescale .* sample_noise(gn, rng_noi=rng_act)   # add noise only in training / choose noise
		# #----------------- Epsilon noise ------------------------------
		# eps = sample_noise(en, rng_noi=rng_act) # add noise only in training / choose noise
		# rng=rand(MersenneTwister(rng_act))
		# if rng > eps
		# 	return act_pred, 0f0
		# elseif rng <= eps
		# 	noise = noisescale .* sample_noise(ou, rng_noi=rng_act)
		noise = noisescale .* randn(MersenneTwister(rng_act), Float32, ACTION_SIZE)
		noise = noiseclamp ? clamp.(noise, -5f-1, 5f-1) : noise #add noise clamp?
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
    a, noise = act(actor, s, noisescale=noisescale, train=train, rng_act=rng_step)
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
		print("Episode: $i | Mean Step Score: $(@sprintf "%9.3f" total_reward[i]) |  $(en.ξ)  |  ")

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
				saveBSON(actor, actor_target, critic, critic_target,
									total_reward, score_mean, best_run, noise_mean,
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
