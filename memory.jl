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

# ---------------------------- Helper Functions --------------------------------
# Stores tuple of state, action, reward, next_state, and done
remember(state, action, reward, next_state, done) =
	push!(memory, [state, action, reward, next_state, done])

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

function saveBSON(actor, actor_target, critic, critic_target)
	actor = actor |> cpu
	actor_target = actor_target |> cpu
	critic = critic |> cpu
	critic_target = critic_target |> cpu

	BSON.@save "out/bson/DDPG_Shems_v12_$(NUM_STEPS)_$(NUM_EP)_$(L1)_$(L2)_$(case)_actor.bson" actor
	BSON.@save "out/bson/DDPG_Shems_v12_$(NUM_STEPS)_$(NUM_EP)_$(L1)_$(L2)_$(case)_actor_target.bson" actor_target
	BSON.@save "out/bson/DDPG_Shems_v12_$(NUM_STEPS)_$(NUM_EP)_$(L1)_$(L2)_$(case)_critic.bson" critic
	BSON.@save "out/bson/DDPG_Shems_v12_$(NUM_STEPS)_$(NUM_EP)_$(L1)_$(L2)_$(case)_critic_target.bson" critic_target
	BSON.@save "out/bson/DDPG_Shems_v12_$(NUM_STEPS)_$(NUM_EP)_$(L1)_$(L2)_$(case)_scores.bson" total_reward score_mean
	return nothing
end

function loadBSON(;scores_only=false)
	if scores_only==true
		BSON.@load "out/bson/DDPG_Shems_v12_$(NUM_STEPS)_$(NUM_EP)_$(L1)_$(L2)_$(case)_scores.bson" total_reward score_mean
		return total_reward, score_mean
	end
	BSON.@load "out/bson/DDPG_Shems_v12_$(NUM_STEPS)_$(NUM_EP)_$(L1)_$(L2)_$(case)_actor.bson" actor
	BSON.@load "out/bson/DDPG_Shems_v12_$(NUM_STEPS)_$(NUM_EP)_$(L1)_$(L2)_$(case)_actor_target.bson" actor_target
	BSON.@load "out/bson/DDPG_Shems_v12_$(NUM_STEPS)_$(NUM_EP)_$(L1)_$(L2)_$(case)_critic.bson" critic
	BSON.@load "out/bson/DDPG_Shems_v12_$(NUM_STEPS)_$(NUM_EP)_$(L1)_$(L2)_$(case)_critic_target.bson" critic_target
	BSON.@load "out/bson/DDPG_Shems_v12_$(NUM_STEPS)_$(NUM_EP)_$(L1)_$(L2)_$(case)_scores.bson" total_reward score_mean
	return actor, actor_target, critic, critic_target, total_reward, score_mean
end
