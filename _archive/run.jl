# Parameters and architecture based on:
# https://github.com/msinto93/DDPG/blob/master/train.py
# https://github.com/JuliaReinforcementLearning/ReinforcementLearningZoo.jl/blob/master/src/experiments/rl_envs/JuliaRL_DDPG_Pendulum.jl
# https://github.com/FluxML/model-zoo/blob/master/contrib/games/differentiable-programming/pendulum/DDPG.jl

# ------------------------------- Action Noise --------------------------------
function sample_noise(ou::OUNoise)
  dx     = ou.θ .* (ou.μ .- ou.X) .* ou.dt
  dx   .+= ou.σ .* sqrt(ou.dt) .* randn(rng, Float32, length(ou.X)) #|> gpu
  return ou.X .+= dx
end

function sample_noise() # Normal distribution
  dx     = randn(rng, Float32, ACTION_SIZE) #|>gpu
  return dx
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
L2_loss(model) = L2_DECAY * sum(x->sum(x.^2), params(model))

loss_crit(model, y, s, a) = Flux.mse(critic(s, a), y) #+ L2_loss(model)

function loss_act(model, s_norm)
  actions = actor(s_norm)
  crit_out = critic(s_norm, actions)
  return -sum(crit_out)  # sum better than mean
end

function replay()
  # retrieve minibatch from replay buffer
  s, a, r, s′ = getData(BATCH_SIZE) # s_mask when with terminal state

  a′ = actor_target(s′)
  v′ = critic_target(s′, a′)
  y = r .+ (γ * v′) #no terminal reward switch off

  # update actor, critic
  update_model!(critic, opt_crit, loss_crit, y, s, a)
  update_model!(actor, opt_act, loss_act, s)
  # update target networks
  update_target!(actor_target, actor; τ = τ)
  update_target!(critic_target, critic; τ = τ)
  return nothing
end

# Choose action according to policy
function action(s_norm; train=true)
	act_pred = actor(s_norm |> gpu) |> cpu
	noise = zeros(ACTION_SIZE)
	if train == true
		##act_pred = act_pred + noise_scale .* sample_noise(ou) #.* noise_scale  # add noise only in training
		noise = sample_noise(ou) .* noise_scale  # add noise only in training #sample_noise(ou)
	end
	return act_pred + noise, mean(noise)
end

function scale_action(action)
	#scale action [-1, 1] to action bounds
	scaled_action = Float32.(ACTION_BOUND_LO +
						(action + ones(ACTION_SIZE)) .*
							0.5 .* (ACTION_BOUND_HI - ACTION_BOUND_LO))
	return scaled_action
end

function episode!(env::Shems; NUM_STEPS=EP_LENGTH["train"], train=true, render=false,
					track=0, rng=0)
  reset!(env, rng=rng)
  local reward_eps=0f0
  local noise_eps=0f0
  local last_step = 1
  local results = Array{Float64}(undef, 0, 25)
  for step=1:NUM_STEPS
	# determine action
	s = copy(env.state)
	s_norm = normalize(s |> gpu)
    a, noise = action(s_norm, train=train)
	scaled_action = scale_action(a)

	# execute action in RL environment
	if track == 0
		r, s′ = step!(env, s, scaled_action)
	elseif track == 1
		r, s′, results_new = step!(env, s, scaled_action, track=track)
		results = vcat(results, results_new)
	elseif track == 2 #rule-based
		r, s′, results_new = step_rule!(env, s)
		results = vcat(results, results_new)
	end
	# render step
	if render == true
		sleep(1f-10)
		gui(plot(env))
		#plot(env) # for gif creation
		#frame(anim) # for gif creation
	end
	# save step in replay buffer
	if train == true
      remember(s_norm, a, r, normalize(s′ |> gpu))  #, finished(env, s′)) # for final state
	  # update network weights
	  replay()
    end
	reward_eps += r
	noise_eps += noise
	last_step = step
	finished(env, s′) && break
  end

  if track == 0
	  return (reward_eps / last_step), (noise_eps / last_step)
  else
      return (reward_eps / last_step), results
  end
end

function run_episodes(env_train::Shems, env_test_eval::Shems, total_reward, score_mean, best_run, noise_mean, noise_scale;
						render=false, track=0)
	reset!(env_train)
	best_score = -10000000
	for i=1:NUM_EP
		total_reward[i], noise_mean[i] = episode!(env_train, train=true, render=render, track=track)
		print("Episode: $i | Mean Step Score: $(@sprintf "%9.3f" total_reward[i]) | ")
		if i % 100 == 1
			idx = ceil(Int32, i/100)
			score_mean[idx,:] = test(env_test_eval)
			print("Mean score $(@sprintf "%9.3f" score_mean[idx,1]) and "*
					"std. deviation $(@sprintf "%9.3f" score_mean[idx,2])  over 100 evaluation episodes | ")
			# save weights for early stopping
			if score_mean[idx,1] > best_score
				# save network weights
				saveBSON(actor, actor_target, critic, critic_target,
									total_reward, score_mean, best_run, noise_mean; idx=i, path="temp")
				# set new best score
				best_score = score_mean[idx,1]
				# save best run
				global best_run = i
			end
		end
		t_elap = round(now()-t_start, Dates.Minute)
		println("Time elapsed: $(t_elap)")
		#global noise_scale *= ϵ
	end
	return nothing
end
