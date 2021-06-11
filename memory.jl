# Parameters and architecture based on:
# https://github.com/msinto93/DDPG/blob/master/train.py
# https://github.com/JuliaReinforcementLearning/ReinforcementLearningZoo.jl/blob/master/src/experiments/rl_envs/JuliaRL_DDPG_Pendulum.jl
# https://github.com/FluxML/model-zoo/blob/master/contrib/games/differentiable-programming/pendulum/DDPG.jl
# --------------------------------- Memory ------------------------------------

function getData(batch_size)
  # Getting data in shape
  minibatch = sample(memory, batch_size)

  s       =   reduce(hcat,getindex.(minibatch,1))  |> gpu
  a       =   reduce(hcat,getindex.(minibatch,2))  |> gpu
  r       =   reduce(hcat,getindex.(minibatch,3))  |> gpu
  s_prime =   reduce(hcat,getindex.(minibatch,4))  |> gpu
 # s_mask  = .!hcat(x[5, :]...) |> gpu #only used for final state

  return s, a, r, s_prime #, s_mask
end

# ---------------------------- Helper Functions --------------------------------
# Stores tuple of state, action, reward, next_state, and done
remember(state, action, reward, next_state) = #, done) =
	push!(memory, [state, action, reward, next_state]) #, done])

# Populate memory with random actions
function populate_memory(env::Shems)
	while length(memory) < MIN_EXP_SIZE
		reset!(env)
		for step=1:EP_LENGTH["train"]
		  s = copy(env.state)
		  # random values between -1 and 1
		  a = Float32.(rand(rng, ACTION_SIZE) .* 2 .- 1)
		  # sclae to action bounds
		  scaled_action = scale_action(a)
		  # execute action in RL environment
		  r, s_prime = step!(env, s, scaled_action)
		  # store tuple in experience buffer
		  remember(s, a, r, s_prime) #, finished(env, s_prime))
		  finished(env, s_prime) && break
		end
	end
	return nothing
end

# --------------------------------- Data preprocessing ------------------------------------
function min_max_buffer(MIN_EXP_SIZE=MIN_EXP_SIZE)
	s, a, r, s_prime = getData(MIN_EXP_SIZE) #s_mask when finite
	return minimum(s, dims=2), maximum(s, dims=2)
end

function normalize(s; s_min=s_min, s_max=s_max)
	return (s .- s_min) ./ (s_max .- s_min .+ 1f-8)
end

function saveBSON(actor, actor_target, critic, critic_target,
					total_reward, score_mean, best_run, noise_mean; idx=NUM_EP, path="")
	actor = actor |> cpu
	actor_target = actor_target |> cpu
	critic = critic |> cpu
	critic_target = critic_target |> cpu

	BSON.@save "out/bson/$(path)/DDPG_Shems_v12_$(EP_LENGTH["train"])_$(NUM_EP)_$(L1)_$(L2)_$(case)_actor_$(idx).bson" actor
	BSON.@save "out/bson/$(path)/DDPG_Shems_v12_$(EP_LENGTH["train"])_$(NUM_EP)_$(L1)_$(L2)_$(case)_actor_target_$(idx).bson" actor_target
	BSON.@save "out/bson/$(path)/DDPG_Shems_v12_$(EP_LENGTH["train"])_$(NUM_EP)_$(L1)_$(L2)_$(case)_critic_$(idx).bson" critic
	BSON.@save "out/bson/$(path)/DDPG_Shems_v12_$(EP_LENGTH["train"])_$(NUM_EP)_$(L1)_$(L2)_$(case)_critic_target_$(idx).bson" critic_target
	BSON.@save "out/bson/$(path)/DDPG_Shems_v12_$(EP_LENGTH["train"])_$(NUM_EP)_$(L1)_$(L2)_$(case)_scores_$(idx).bson" total_reward score_mean best_run noise_mean
	return nothing
end

function loadBSON(;idx=NUM_EP, scores_only=false, path="")
	if scores_only==true
		BSON.@load "out/bson/DDPG_Shems_v12_$(EP_LENGTH["train"])_$(NUM_EP)_$(L1)_$(L2)_$(case)_scores_$(idx).bson" total_reward score_mean best_run noise_mean
		return total_reward |> gpu, score_mean |> gpu, best_run |> gpu, noise_mean |> gpu
	end
	BSON.@load "out/bson/$(path)/DDPG_Shems_v12_$(EP_LENGTH["train"])_$(NUM_EP)_$(L1)_$(L2)_$(case)_actor_$(idx).bson" actor
	BSON.@load "out/bson/$(path)/DDPG_Shems_v12_$(EP_LENGTH["train"])_$(NUM_EP)_$(L1)_$(L2)_$(case)_actor_target_$(idx).bson" actor_target
	BSON.@load "out/bson/$(path)/DDPG_Shems_v12_$(EP_LENGTH["train"])_$(NUM_EP)_$(L1)_$(L2)_$(case)_critic_$(idx).bson" critic
	BSON.@load "out/bson/$(path)/DDPG_Shems_v12_$(EP_LENGTH["train"])_$(NUM_EP)_$(L1)_$(L2)_$(case)_critic_target_$(idx).bson" critic_target
	BSON.@load "out/bson/$(path)/DDPG_Shems_v12_$(EP_LENGTH["train"])_$(NUM_EP)_$(L1)_$(L2)_$(case)_scores_$(idx).bson" total_reward score_mean best_run noise_mean
	return actor |> gpu, actor_target |> gpu, critic |> gpu, critic_target |> gpu, total_reward |> gpu, score_mean |> gpu, best_run |> gpu, noise_mean |> gpu
end
