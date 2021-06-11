#include("out/$(ENV["JOB_ID"])-input.jl") # contains the input data
include("input.jl") # contains the input data
include("memory.jl") # contains all data buffer functions
include("run.jl") # contains all training functions
include("testing.jl") # contains testing functions
include("plotting_saving.jl") # contains all ploting and rendering functions

populate_memory(env_dict["train"])
# initialization for normalization
s_min, s_max = min_max_buffer(MIN_EXP_SIZE)
# ------------------------------ Training --------------------------------------
total_reward = zeros(Float32, NUM_EP)
noise_mean = zeros(Float32, NUM_EP)
best_run = 0
score_mean = zeros(ceil(Int32, NUM_EP/100),2)

# ------------------------- train ---------------------------------------
if train == true
	t_start = now()
	print("Max steps: $(EP_LENGTH["train"]), Max episodes: $(NUM_EP), Layer 1: $(L1) nodes, Layer 2: $(L2) nodes, ")
	println("Case: $(case), Time to start: $(round(t_start - start_time, Dates.Minute))")
	run_episodes(env_dict["train"], env_dict[run], total_reward, score_mean, best_run, noise_mean, noise_scale, render=render)
	# ------------------------- Save results ---------------------------------------
	saveBSON(actor, actor_target, critic, critic_target, total_reward, score_mean, best_run, noise_mean)
end

# ------------------------- render evaluation ---------------------------------------
if render == true
	actor, actor_target, critic, critic_target, total_reward, score_mean, best_run, noise_mean = loadBSON(idx=idx)
	#gr()
	#global anim = Animation() # gif
	inference(env_dict[run]; render=true, track=0, idx=idx)
	#gif(anim, "analysis/DDPG_Shems_v12_$(NUM_STEPS)_$(NUM_EP)_$(L1)_$(L2)_$(case).gif", fps=5) # gif
end

# ------------------------- track evaluation ---------------------------------------
if track == 1 # track last and best training run
	# track last episode weights
	actor, actor_target, critic, critic_target, total_reward, score_mean, best_run, noise_mean = loadBSON()
	inference(env_dict[run]; render=false, track=track)
	write_to_tracker_file()
	best_eval = best_run
	# track best episode weights
	actor, actor_target, critic, critic_target, total_reward, score_mean, best_run, noise_mean =
			loadBSON(idx=best_eval, path="temp")
	inference(env_dict[run]; render=false, track=track, idx=best_eval)
	write_to_tracker_file(idx=best_eval)
elseif track == 2 #rule-based
	inference(env_dict[run]; render=false, track=track, idx=-1)
	write_to_tracker_file(idx=-1)

end

# ------------------------- plot scores ---------------------------------------
if plot_result == true
	total_reward, score_mean, best_run, noise_mean = loadBSON(scores_only=true)
	println("train (last $(round(Int, length(total_reward)/20)+1))=
			$(mean(total_reward[end-round(Int, length(total_reward)/20):end]))")
	println("eval (last $(round(Int, size(score_mean)[1]/10)+1))=
			$(mean(score_mean[end-round(Int, size(score_mean)[1]/10):end,1]))")
	plot_scores(ymin = -20)
	plot_scores(ymin = -1)
end
