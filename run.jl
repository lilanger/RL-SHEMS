# Parameters and architecture based on:
# https://github.com/msinto93/DDPG/blob/master/train.py
# https://github.com/JuliaReinforcementLearning/ReinforcementLearningZoo.jl/blob/master/src/experiments/rl_envs/JuliaRL_DDPG_Pendulum.jl
# https://github.com/FluxML/model-zoo/blob/master/contrib/games/differentiable-programming/pendulum/DDPG.jl

# ------------------------------- Action Noise --------------------------------
function sample_noise(ou::OUNoise)
  dx     = ou.θ * (ou.μ .- ou.X) * ou.dt
  dx   .+= ou.σ * sqrt(ou.dt) * randn(rng, Float32, length(ou.X))
  return ou.X .+= dx
end

# ---------------------- Param Update Functions --------------------------------
function update_target!(target, model; τ = 1f0)
  for (p_t, p_m) in zip(params(target), params(model))
    p_t .= (1f0 - τ) * p_t .+ τ * p_m
  end
end

function update_model!(model, opt, loss, inp...)
  grads = gradient(()->loss(inp...), params(model))
  update!(opt, params(model), grads)
end

# ---------------------------------- Training ----------------------------------
# Losses
# function L2_loss(model)
#   para = params(model)
#   l2_loss = sum(map(p->sum(p.^2), para))
#   return L2_DECAY * l2_loss
# end

loss_crit(y, s, a) = Flux.mse(critic(s, a), y) #+ L2_loss(critic) # L2 loss has huge pos. impact

function loss_act(s_norm)
  actions = actor(s_norm)
  crit_out = critic(s_norm, actions)
  return -sum(crit_out)  # sum better than mean
end

function replay()
  s, a, r, s′, s_mask = getData()

  a′ = actor_target(s′)
  v′ = critic_target(s′, a′)
  y = r .+ (γ * v′ .* s_mask)	# set v′ to 0 where s_ is terminal state

  update_model!(critic, opt_crit, loss_crit, y, s, a)
  update_model!(actor, opt_act, loss_act, s)

  update_target!(actor_target, actor, τ = τ)
  update_target!(critic_target, critic, τ = τ)
  return nothing
end

# Choose action according to policy
function action(s_norm; train=true)
	act_pred = actor(s_norm) |> cpu
	act_pred += train .* noise_scale .* sample_noise(ou) # add noise only in training
	return act_pred
end

function episode!(env::Shems; train=true, render=false, track=false)
  reset!(env)
  local reward_eps=0f0
  local last_step = 1
  local results = Array{Float64}(undef, 0, 22)
  for step=1:NUM_STEPS
	s = copy(env.state)
	s_norm = normalize(s |> cpu)
    a = action(s_norm |> gpu, train=train) |> cpu
	scaled_action = Float32.(ACTION_BOUND_LO .+
						(a .+ ones(ACTION_SIZE)) .*
							0.5 .* (ACTION_BOUND_HI .- ACTION_BOUND_LO)) #scale to action range
	if track == false
		r, s_prime = step!(env, s, scaled_action)
	elseif track ==true
		r, s_prime, results_new = step!(env, s, scaled_action, track=track)
		results = vcat(results, results_new)
	end
	reward_eps += r
	finished(env, s_prime) && break

	if render == true
		#sleep(1f-2)
		#gui(plot(env))
		plot(env)
		frame(anim)
	end

	if train == true
      remember(s_norm, a, r, normalize(s_prime |> cpu), finished(env, s_prime))
	  if step % UPDATE_EVERY == 0
		  replay()
	  end
    end

	last_step = step
  end

  track == false ? (return (reward_eps / last_step)) : (return (reward_eps / last_step), results)
end

function run_episodes(env::Shems, env_eval::Shems, total_reward, score_mean, noise_scale;
						render=false, track=false)
	reset!(env)
	for i=1:NUM_EP
		total_reward[i] = episode!(env, train=true, render=render, track=track)
		print("Episode: $i | Mean Step Score: $(@sprintf "%9.3f" total_reward[i]) | ")
		if i % 10 == 1
			idx = ceil(Int32, i/10)
			score_mean[idx] = test(env_eval)
			print("Mean score over 100 test episodes: $(@sprintf "%9.3f" score_mean[idx]) | ")
		end
		t_elap = round(now()-t_start, Dates.Minute)
		println("Time elapsed: $(t_elap)")
		noise_scale = ϵ * noise_scale
	end
	return total_reward, score_mean
end
