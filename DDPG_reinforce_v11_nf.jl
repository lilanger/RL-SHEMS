include("input.jl")
include("functions.jl")

populate_memory(env::Shems)
# initialization for normalization
s_min, s_max = min_max_buffer(MIN_EXP_SIZE)

# ------------------------------ Training --------------------------------------
total_reward = zeros(Float32, NUM_EP)
score_mean = zeros(Float32, ceil(Int32, NUM_EP/10))
if render_test == false
	t_start = now()
	print("Max steps: $(NUM_STEPS), Max episodes: $(NUM_EP), Layer 1: $(L1) nodes, Layer 2: $(L2) nodes, ")
	println("Case: $(case), Time to start: $(round(t_start - start_time, Dates.Minute))")
	run_episodes(env, total_reward, score_mean, noise_scale, s_min, s_max)

	# ------------------------- Save results ---------------------------------------
    if save_result == true
		actor = cpu(actor)
		actor_target = cpu(actor_target)
		critic = cpu(critic)
		critic_target = cpu(critic_target)

		BSON.@save "out/bson/DDPG_Shems_v11_$(NUM_STEPS)_$(NUM_EP)_$(L1)_$(L2)_$(case)_actor.bson" actor
		BSON.@save "out/bson/DDPG_Shems_v11_$(NUM_STEPS)_$(NUM_EP)_$(L1)_$(L2)_$(case)_actor_target.bson" actor_target
		BSON.@save "out/bson/DDPG_Shems_v11_$(NUM_STEPS)_$(NUM_EP)_$(L1)_$(L2)_$(case)_critic.bson" critic
		BSON.@save "out/bson/DDPG_Shems_v11_$(NUM_STEPS)_$(NUM_EP)_$(L1)_$(L2)_$(case)_critic_target.bson" critic_target
		BSON.@save "out/bson/DDPG_Shems_v11_$(NUM_STEPS)_$(NUM_EP)_$(L1)_$(L2)_$(case)_scores.bson" total_reward score_mean
	end

	if plot_result == true
		plot_scores()
	end

elseif render_test == true
	# ------------------------ Load and render resulting behavior --------------------
	gr()
	render_results(env; NUM_STEPS=NUM_STEPS, NUM_EP=NUM_EP, case=case, plot_result=plot_result)
end
