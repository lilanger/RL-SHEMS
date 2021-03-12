# Parameters and architecture based on:
# https://github.com/msinto93/DDPG/blob/master/train.py
# https://github.com/JuliaReinforcementLearning/ReinforcementLearningZoo.jl/blob/master/src/experiments/rl_envs/JuliaRL_DDPG_Pendulum.jl
# https://github.com/FluxML/model-zoo/blob/master/contrib/games/differentiable-programming/pendulum/DDPG.jl

function plot_scores(;ymin=Inf, total_reward=total_reward, score_mean=score_mean)
	scatter(1:NUM_EP, [total_reward], label="train", left_margin = 12mm,
			markershape=:circle, markersize=3, markerstrokewidth=0.1,
			legend=:bottomright, ylim=(ymin, 0.5))
	scatter!(1:10:NUM_EP, score_mean, label="test", left_margin = 12mm,
			markershape=:circle, legend=:bottomright, markersize=3, markerstrokewidth=0.1)
	savefig("out/fig/DDPG_Shems_v12_$(NUM_STEPS)_$(NUM_EP)_$(L1)_$(L2)_$(case)_$(ymin)")
end

function write_to_results_file(results)
    date=Date(now());
    CSV.write("out/$(date)_results_v12_$(NUM_STEPS)_$(NUM_EP)_$(L1)_$(L2)_$(case).csv",
			DataFrame(results, :auto), header=["Temp_FH", "Vol_HW", "Soc_B",
			"V_HW_plus", "V_HW_minus", "T_FH_plus", "T_FH_minus",
			"profits", "COP_FH","COP_HW","PV_DE", "B_DE", "GR_DE", "PV_B", "PV_GR",
			"PV_HP","GR_HP", "B_HP", "HP_FH", "HP_HW","hour", "index"]);
    return nothing
end
