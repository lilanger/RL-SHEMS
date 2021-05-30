function plot_scores(;ymin=Inf, total_reward=total_reward, score_mean=score_mean)
	scatter(1:NUM_EP, [total_reward], label="train", left_margin = 12Plots.mm,
			markershape=:circle, markersize=2, markerstrokewidth=0.1,
			legend=:bottomright, ylim=(ymin, 0.5))
	plot!(1:100:NUM_EP, score_mean[:,1], label="test", left_margin = 12Plots.mm,
			markershape=:circle, legend=:bottomright, markersize=2, markerstrokewidth=0.1)
	plot!(1:100:NUM_EP, score_mean[:,1] + score_mean[:,2], fillrange=score_mean[:,1] - score_mean[:,2], colour =:orange, alpha=0.2);
	savefig("out/fig/DDPG_Shems_v12_$(EP_LENGTH["train"])_$(NUM_EP)_$(L1)_$(L2)_$(case)_$(ymin)")
end

function write_to_results_file(results; idx=NUM_EP)
    date=Date(now());
    CSV.write("out/$(date)_results_v12_$(EP_LENGTH["train"])_$(NUM_EP)_$(L1)_$(L2)_$(case)_$(idx).csv",
			DataFrame(results, :auto), header=["Temp_FH", "Vol_HW", "Soc_B",
			"T_FH_plus", "T_FH_minus", "V_HW_plus", "V_HW_minus",
			"profits", "COP_FH","COP_HW","PV_DE", "B_DE", "GR_DE", "PV_B", "PV_GR",
			"PV_HP","GR_HP", "B_HP", "HP_FH", "HP_HW","index", "B", "HP"]);
    return nothing
end


function write_to_tracker_file(idx=NUM_EP)
	time=now();
	date=Date(time);
	results = CSV.read("out/$(date)_results_v12_$(EP_LENGTH["train"])_$(NUM_EP)_$(L1)_$(L2)_$(case)_$(idx).csv", DataFrame)
	Tracker = CSV.read("out/Tracker.csv", DataFrame)
	Tracker = vcat(Tracker, [time, NUM_EP, L1, L2, BATCH_SIZE, MEM_SIZE, MIN_EXP_SIZE,
								season, case, idx,
								sum(results[!, :T_fh_plus]), sum(results[!, :T_fh_minus]),
								sum(results[!, :V_hw_plus]), sum(results[!, :V_hw_minus]),
								sum(results[!, :reward])])

    CSV.write("out/Tracker.csv", Tracker);
    return nothing
