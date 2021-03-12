include("input.jl") # contains the input data
include("memory.jl") # contains all data buffer functions
include("run.jl") # contains all training functions
include("testing.jl") # contains testing functions
include("plotting.jl") # contains all ploting and rendering functions

populate_memory(env)
# initialization for normalization
s_min, s_max = min_max_buffer(MIN_EXP_SIZE)
# ------------------------------ Training --------------------------------------
total_reward = zeros(Float32, NUM_EP)
score_mean = zeros(Float32, ceil(Int32, NUM_EP/10))

# ------------------------- train ---------------------------------------
if train == true
	t_start = now()
	print("Max steps: $(NUM_STEPS), Max episodes: $(NUM_EP), Layer 1: $(L1) nodes, Layer 2: $(L2) nodes, ")
	println("Case: $(case), Time to start: $(round(t_start - start_time, Dates.Minute))")
	run_episodes(env, env_eval, total_reward, score_mean, noise_scale, render=render)
	# ------------------------- Save results ---------------------------------------
	saveBSON(actor, actor_target, critic, critic_target)
else
	# ------------------------- render evaluation ---------------------------------------
	if render == true
		actor, actor_target, critic, critic_target, total_reward, score_mean = loadBSON()
		gr()
		inference(render=render, track=track)
	elseif track == true
		actor, actor_target, critic, critic_target, total_reward, score_mean = loadBSON()
		inference(render=render, track=track)
	end
end

# ------------------------- plot scores ---------------------------------------
if plot_result == true
	total_reward, score_mean = loadBSON(scores_only=true)
	println("train (last $(round(Int, length(total_reward)/20)+1))= $(mean(total_reward[end-round(Int, length(total_reward)/20):end]))")
	println("eval (last $(round(Int, length(score_mean)/10)+1))= $(mean(score_mean[end-round(Int, length(score_mean)/10):end]))")
	plot_scores(ymin = -20)
	plot_scores(ymin = -1)
end
