try 
	# run batch on cluster
	println("Use cluster setup.")
	include("out/input/$(ENV["JOB_ID"])--input.jl") # contains the input data
	sleep(200) # sleep for the input to load before calling algo
catch 
	# run single file
	println("Use single run setup.")
	include("input.jl") # contains the input data
end

include("algorithms/$(algo).jl") # contains all training functions
include("src/memory_plotting_saving.jl") # contains all ploting and rendering functions
#-------
populate_memory(env_dict["train"], rng=rng_run)
# initialization for normalization
s_min, s_max = min_max_buffer(MIN_EXP_SIZE, rng_mm=rng_run) |> gpu
# ------------------------------ Training --------------------------------------
total_reward = zeros(Float32, NUM_EP)
noise_mean = zeros(Float32, NUM_EP)
best_run = 0
score_mean = zeros(ceil(Int32, NUM_EP/test_every))

# ------------------------- train ---------------------------------------
if train == true
	t_start = now()
	print("Max steps: $(EP_LENGTH["train"]), Max episodes: $(NUM_EP), Layer 1: $(L1) nodes, Layer 2: $(L2) nodes, ")
	println("Case: $(case), Time to start: $(round(t_start - start_time, Dates.Minute))")
	run_episodes(env_dict["train"], env_dict["eval"], total_reward, score_mean, best_run, noise_mean,
					test_every, render,  rng_run, track=0)
	# ------------------------- Save results ---------------------------------------
	saveBSON(actor, total_reward, score_mean, best_run, noise_mean,
				rng=rng_run)
end

# ------------------------- render evaluation ---------------------------------------
if render == true && train != true
	actor,total_reward, score_mean, best_run, noise_mean =
			loadBSON(idx=idx, rng=rng_run)
	#gr()
	#global anim = Animation() # gif
	inference(env_dict[run]; render=true, track=0, idx=idx)
	#gif(anim, "analysis/DDPG_Shems_v12_$(NUM_STEPS)_$(NUM_EP)_$(L1)_$(L2)_$(case).gif", fps=5) # gif
end

# ------------------------- plot scores ---------------------------------------
if plot_result == true
	total_reward, score_mean, best_run, noise_mean = loadBSON(scores_only=true, rng=rng_run) |> cpu
	println("train (last $(round(Int, length(total_reward)/20)+1))=
			$(mean(total_reward[end-round(Int, length(total_reward)/20):end]))")
	println("eval (last $(round(Int, size(score_mean)[1]/10)+1))=
			$(mean(score_mean[end-round(Int, size(score_mean)[1]/10):end,1]))")
	plot_scores(ymin = -2, rng=rng_run)
end

if plot_all == true
	if seed_run == num_seeds
		# delay to be sure all runs are done
		sleep(WAIT[season, algo]) 

  		score_mean_all = zeros(Float32, (ceil(Int32, NUM_EP/test_every), num_seeds)) |> cpu
		for i in 1:num_seeds
			test_rng_run = parse(Int, string(seed_ini)*string(i))
			score_mean_all[:,i] = loadBSON(scores_only=true, rng=test_rng_run)[2] |> cpu
		end
		plot_all_scores(ymin = -50, score_mean=score_mean_all) #ymin = -10, 
	end
end


# ------------------------- track evaluation ---------------------------------------
if track == 1 # track last and best training run
	if seed_run == num_seeds
		for i in 1:num_seeds
			test_rng_run = parse(Int, string(seed_ini)*string(i))
			# track last episode weights
			ac, tr, sm, best_eval, nm = loadBSON(rng=test_rng_run)
			global actor = deepcopy(ac) |> gpu
			inference(env_dict[run], render=false, track=track, rng_inf=test_rng_run)
			write_to_tracker_file(rng=test_rng_run)

			# track best episode weights
		    ac = loadBSON(idx=best_eval, path="temp", rng=test_rng_run)[1]
			global actor = deepcopy(ac) |> gpu
			inference(env_dict[run]; render=false, track=track, idx=best_eval, rng_inf=test_rng_run, best=true)
			write_to_tracker_file(idx=best_eval, rng=test_rng_run, best=true)
		end
	end

elseif track < 0 #rule-based
	inference(env_dict[run]; render=false, track=track, idx=track)
	write_to_tracker_file(idx=track, rng=track)
end
