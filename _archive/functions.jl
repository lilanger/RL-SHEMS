# Parameters and architecture based on:
# https://github.com/msinto93/DDPG/blob/master/train.py
# https://github.com/JuliaReinforcementLearning/ReinforcementLearningZoo.jl/blob/master/src/experiments/rl_envs/JuliaRL_DDPG_Pendulum.jl
# https://github.com/FluxML/model-zoo/blob/master/contrib/games/differentiable-programming/pendulum/DDPG.jl
# --------------------------------- Memory ------------------------------------
function getData(batch_size = BATCH_SIZE)
  # Getting data in shape
  minibatch = sample(memory, batch_size)
  x = hcat(minibatch...)

  s       =   hcat(x[1, :]...) |> gpu
  a       =   hcat(x[2, :]...) |> gpu
  r       =   hcat(x[3, :]...) |> gpu
  s_prime =   hcat(x[4, :]...) |> gpu
  s_mask  = .!hcat(x[5, :]...) |> gpu

  return s, a, r, s_prime, s_mask
end

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
function L2_loss(model)
  l2_loss = sum(map(p->sum(p.^2), params(model)))
  return L2_DECAY * l2_loss
end

loss_crit(y, s, a) = Flux.mse(critic(s, a), y) + L2_loss(critic) # L2 loss has huge pos. impact

function loss_act(s_norm)
  actions = actor(s_norm)
  crit_out = critic(s_norm, actions)
  return -sum(crit_out)  # sum better than mean
end

function replay(s_min, s_max)
  s, a, r, s′, s_mask = getData()
  # adjust normalization values
  s_min = (1f0 - τ) * s_min .+ τ * minimum(s |> cpu, dims=2)
  s_max = (1f0 - τ) * s_max .+ τ * maximum(s |> cpu, dims=2)

  a′ = actor_target(s′)
  v′ = critic_target(s′, a′)
  y = r .+ (γ * v′ .* s_mask)	# set v′ to 0 where s_ is terminal state

  update_model!(critic, opt_crit, loss_crit, y, s, a)
  update_model!(actor, opt_act, loss_act, s)

  update_target!(actor_target, actor, τ = τ)
  update_target!(critic_target, critic, τ = τ)
  return s_min, s_max
end

# ---------------------------- Helper Functions --------------------------------
# Stores tuple of state, action, reward, next_state, and done
remember(state, action, reward, next_state, done) =
	push!(memory, [state, action, reward, next_state, done])

# Choose action according to policy
function action(s_norm; train=true)
	act_pred = actor(s_norm) |> cpu
	act_pred += train .* noise_scale .* sample_noise(ou) # add noise only in training
	return act_pred
end

function episode!(env::Shems, s_min, s_max; train=true, render=false)
  reset!(env)
  local reward_eps=0f0
  local last_step = 1
  for step=1:NUM_STEPS
	s = copy(env.state)
	#println(s)
	s_norm = normalize(s |> cpu)
	#println(s_norm)
    a = action(s_norm |> gpu, train=train) |> cpu
	#println(a)
	scaled_action = Float32.(ACTION_BOUND_LO .+
						(a .+ ones(ACTION_SIZE)) .*
							0.5 .* (ACTION_BOUND_HI .- ACTION_BOUND_LO)) #scale to action range
	#println(scaled_action)
	r, s_prime = step!(env, s, scaled_action)
	#println(r)
	#println(s_prime)
	#println()
	reward_eps += r
	finished(env, s_prime) && break
	if render == true
		sleep(1f-1)
		gui(plot(env))
	end
	if train == true
      remember(s_norm, a, r, normalize(s_prime |> cpu), finished(env, s_prime))
	  if step % UPDATE_EVERY == 0
		  s_min, s_max = replay(s_min, s_max)
	  end
    #elseif train == false
	  #println(s)
	  # println(scaled_action)
	  # println(r)
	  # println(s_prime)
    end
	last_step = step
  end
  return (reward_eps / last_step)
end
# -------------------------------- Testing -------------------------------------

# Returns average score (per step) over 100 episodes
function test(env::Shems,s_min, s_max; render=false)
  reward = 0f0
  for e=1:100
  	reward += episode!(env, s_min, s_max, train=false, render=render)
  end
  return (reward / 100.)
end

function plot_scores(;total_reward=total_reward, score_mean=score_mean)
	scatter(1:NUM_EP, [total_reward], label="train", left_margin = 12mm,
			markershape=:circle, markersize=3, markerstrokewidth=0.1,
			legend=:bottomright)
	scatter!(1:10:NUM_EP, score_mean, label="test", left_margin = 12mm,
			markershape=:circle, legend=:bottomright, markersize=3, markerstrokewidth=0.1)
	savefig("out/fig/DDPG_Shems_v11_$(NUM_STEPS)_$(NUM_EP)_$(L1)_$(L2)_$(case)")
end

# Populate memory with random actions
function populate_memory(env::Shems)
	while length(memory) < MIN_EXP_SIZE
		reset!(env)
		for step=1:NUM_STEPS
		  s = copy(env.state)
		  a = Float32.(rand(rng, ACTION_SIZE) .* 2 .- 1) # random values between -1 and 1
		  scaled_action = Float32.(ACTION_BOUND_LO .+
		  					(a .+ ones(ACTION_SIZE)) .*
								0.5 .* (ACTION_BOUND_HI .- ACTION_BOUND_LO)) #scale to action range
		  r, s_prime = step!(env, s, scaled_action)
		  remember(s, a, r, s_prime, finished(env, s_prime))
		  finished(env, s_prime) && break
		end
	end
	return nothing
end

# --------------------------------- Data preprocessing ------------------------------------
function min_max_buffer(MIN_EXP_SIZE=MIN_EXP_SIZE)
	s, a, r, s_prime, s_mask = getData(MIN_EXP_SIZE)
	return minimum(s |> cpu , dims=2), maximum(s |> cpu, dims=2)
end

function normalize(s; s_min=s_min, s_max=s_max)
	s_norm = (s .- s_min) ./ (s_max .- s_min .+ 1f-8)
	return s_norm
end

function run_episodes(env::Shems, total_reward, score_mean, noise_scale, s_min, s_max)
	reset!(env)
	for i=1:NUM_EP
		total_reward[i] = episode!(env, s_min, s_max, train=true, render=true)
		print("Episode: $i | Mean Step Score: $(@sprintf "%9.3f" total_reward[i]) | ")
		if i % 10 == 1
			idx = ceil(Int32, i/10)
			score_mean[idx] = test(env, s_min, s_max)
			print("Mean score over 100 test episodes: $(@sprintf "%9.3f" score_mean[idx]) | ")
		end
		t_elap = round(now()-t_start, Dates.Minute)
		println("Time elapsed: $(t_elap)")
		noise_scale = ϵ * noise_scale
	end
	return total_reward, score_mean
end

function render_results(env::Shems; NUM_STEPS=NUM_STEPS, NUM_EP=NUM_EP, L1=L1, L2=L2, case=case, plot_result=plot_result)
	BSON.@load "out/bson/DDPG_Shems_v11_$(NUM_STEPS)_$(NUM_EP)_$(L1)_$(L2)_$(case)_actor.bson" actor
	BSON.@load "out/bson/DDPG_Shems_v11_$(NUM_STEPS)_$(NUM_EP)_$(L1)_$(L2)_$(case)_actor_target.bson" actor_target
	BSON.@load "out/bson/DDPG_Shems_v11_$(NUM_STEPS)_$(NUM_EP)_$(L1)_$(L2)_$(case)_critic.bson" critic
	BSON.@load "out/bson/DDPG_Shems_v11_$(NUM_STEPS)_$(NUM_EP)_$(L1)_$(L2)_$(case)_critic_target.bson" critic_target
	if plot_result == true
		BSON.@load "out/bson/DDPG_Shems_v11_$(NUM_STEPS)_$(NUM_EP)_$(L1)_$(L2)_$(case)_scores.bson" total_reward score_mean
		plot_scores()
	end
	test(env, render=render_test)
	return nothing
end
