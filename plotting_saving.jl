function plot_scores(;ymin=Inf, total_reward=total_reward, score_mean=score_mean, noise_mean=noise_mean)
	# plot training results
	scatter(1:NUM_EP, [total_reward], label="train",
			markershape=:circle, markersize=2, markerstrokewidth=0.1,
			legend=:bottomright, ylim=(ymin, 0.5), colour = :darkturquoise);
	# plot training moving average
	plot!(1:NUM_EP, [mean(total_reward[1:i]) for i in 1:length(total_reward)],
			label="train (moving average)",
			colour = :darkturquoise, markerstrokewidth=0.2);
	# plot training moving average
	plot!(1:NUM_EP, [noise_mean],
					label="noise",
					colour = :magenta, markerstrokewidth=0.2);
	# plot test/evaluation results mean
	plot!(1:100:NUM_EP, score_mean[:,1], label="test (mean)",
			markershape=:circle, colour =:orange, markersize=3, markerstrokewidth=0.2);
	# plot test/evaluation 95% confodence inteval (n=100)
	plot!(1:100:NUM_EP, score_mean[:,1] + 1.96 .* score_mean[:,2],
			fillrange =  score_mean[:,1] - 1.96 .* score_mean[:,2],
			label="test (95% confidence interval)", colour =:orange, alpha=0.2);
	savefig("out/fig/DDPG_Shems_v12_$(run)_$(EP_LENGTH["train"])_$(NUM_EP)_$(L1)_$(L2)_$(case)_$(ymin).png")
end

function write_to_results_file(results; idx=NUM_EP)
    date=Date(now());
	if idx == NUM_EP
    CSV.write("out/$(date)_$(run)_results_v12_$(EP_LENGTH["train"])_$(NUM_EP)_$(L1)_$(L2)_$(case)_$(idx).csv",
			DataFrame(results, :auto), header=["Temp_FH", "Vol_HW", "Soc_B",
			"T_FH_plus", "T_FH_minus", "V_HW_plus", "V_HW_minus",
			"profits", "comfort", "abort", "COP_FH","COP_HW","PV_DE", "B_DE", "GR_DE", "PV_B", "PV_GR",
			"PV_HP","GR_HP", "B_HP", "HP_FH", "HP_HW","index", "B", "HP"]);
	elseif idx < 0
	CSV.write("out/$(date)_$(run)_results_$(case)_rule_$(idx).csv",
				DataFrame(results, :auto), header=["Temp_FH", "Vol_HW", "Soc_B",
				"T_FH_plus", "T_FH_minus", "V_HW_plus", "V_HW_minus",
				"profits", "comfort", "abort", "COP_FH","COP_HW","PV_DE", "B_DE", "GR_DE", "PV_B", "PV_GR",
				"PV_HP","GR_HP", "B_HP", "HP_FH", "HP_HW","index", "B", "HP"]);
	else
	CSV.write("out/$(date)_$(run)_results_v12_$(EP_LENGTH["train"])_$(NUM_EP)_$(L1)_$(L2)_$(case)_best.csv",
			DataFrame(results, :auto), header=["Temp_FH", "Vol_HW", "Soc_B",
			"T_FH_plus", "T_FH_minus", "V_HW_plus", "V_HW_minus",
			"profits", "comfort", "abort", "COP_FH","COP_HW","PV_DE", "B_DE", "GR_DE", "PV_B", "PV_GR",
			"PV_HP","GR_HP", "B_HP", "HP_FH", "HP_HW","index", "B", "HP"]);
	end
    return nothing
end


function write_to_tracker_file(;idx=NUM_EP)
	time=now();
	date=Date(now());
	if idx == NUM_EP
		results = CSV.read("out/$(date)_$(run)_results_v12_$(EP_LENGTH["train"])_$(NUM_EP)_$(L1)_$(L2)_$(case)_$(idx).csv", DataFrame)
	elseif idx < 0
		results = CSV.read("out/$(date)_$(run)_results_$(case)_rule_$(idx).csv", DataFrame)
	else
		results = CSV.read("out/$(date)_$(run)_results_v12_$(EP_LENGTH["train"])_$(NUM_EP)_$(L1)_$(L2)_$(case)_best.csv", DataFrame)
	end
	Tracker = Matrix(CSV.read("out/Tracker.csv", DataFrame, header=true))
	Tracker = vcat(Tracker, [time NUM_EP L1  L2  BATCH_SIZE  MEM_SIZE  MIN_EXP_SIZE season  run case  idx [
								sum(results[!, :T_FH_plus])]  sum(results[!, :T_FH_minus]) [
								sum(results[!, :V_HW_plus])]  sum(results[!, :V_HW_minus]) [
								sum(results[!, :profits])] sum(results[!, :comfort]) sum(results[!, :abort])])

    CSV.write("out/Tracker.csv", DataFrame(Tracker,:auto), header=["time", "NUM_EP", "L1", "L2", "BATCH_SIZE", "MEM_SIZE",
								"MIN_EXP_SIZE","season", "run", "case", "idx", "T_FH_plus", "T_FH_minus", "V_HW_plus",
								"V_HW_minus", "profits", "comfort", "abort"]);
    return nothing
end
