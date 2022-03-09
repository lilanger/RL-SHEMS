function read_data(season; tariff="fix", job_id = "1063970", case="$(season)_L2_nns_abort-0", network="Opt", stop="final", NUM_EP = 3001, run="eval", seed=1)
    
    version = "v12"
    NUM_STEPS = 24
    
    Input_df = CSV.read("../data/$(season)_$(run)_$(tariff).csv", DataFrame);
    
    L1 = 300
    L2 = 600  
        
    if network == "Opt"
       Flow_df = CSV.read("../benchmarks/MPC/results/210705_results_test_all_1.0.csv", DataFrame);
       Data_df = hcat(Flow_df[1:end-1, 1:end-4], Input_df[1:end-1, :]) # cut index + hour + last input (no action)
       return Data_df
            
     elseif network == "Rule-based"
       #Flow_df = CSV.read("benchmarks/Rule-based/$(case).csv", DataFrame);
       Flow_df = CSV.read("../out/tracker/$(case).csv",DataFrame);
        Data_df = hcat(Flow_df[1:end, :], Input_df[1:end-1, :]) # cut index + hour + last input (no action)
       return Data_df
    end

    Flow_df=CSV.read("../out/tracker/$(job_id)_$(run)_results_$(version)_$(NUM_STEPS)_$(NUM_EP)_"*
                        "$(L1)_$(L2)_$(case)_123$(seed)_$(NUM_EP).csv",DataFrame);
    #println(size(Flow_df),  size(Input_df))
    Data_df = hcat(Flow_df[1:end, :], Input_df[1:end-1, :]) # cut index + hour + last input (no action)
    return Data_df
end

function bar_target_B(Data_df, start, yaxis, title)
    # SOCs
    @df Data_df[start:start+23,:] bar(
        100 .*[:Soc_B]/ 10, xticks=0:6:24, yticks=0:25:100, tickfontsize=10, ylim=(0,100), colour =:purple,
        label=["Soc_B"], legend=false, legendfontsize=8, title="$(title)", titlefontsize=11,
        titlelocation=:left, linewidth= 0, alpha=0.5, left_margin=3Plots.mm, bottom_margin=4Plots.mm );
  
        # Targets
    @df Data_df[start:start+23,:] scatter!(
        100 .*[:B_tar], colour =:purple, marker=(:circle, stroke(.5)), markersize = 3,
        label=["B_tar"], legend=false, legendfontsize=8);

    annotate!(3, 95, text("$(Data_df[start,:month])/$(Data_df[start,:day])", 10, "serif-roman"))
    yaxis!("$(yaxis)", font(10, "serif-roman"), tickfontsize=10)
    xaxis!("time", font(10, "serif-roman"))
end

function bar_target_HP(Data_df, start, yaxis, title)
    # SOCs
    @df Data_df[start:start+23,:] groupedbar(
        100 .*(([:Temp_FH :Vol_HW] .-[19 20]) ./ [5 160]), xticks=0:6:24, yticks=0:25:100, tickfontsize=10, ylim=(0,100),
        colour =[:firebrick :steelblue],
        label=["Temp_FH" "Vol_HW"], legend=false, legendfontsize=8, title="$(title)", titlefontsize=11,
        titlelocation=:left, linewidth= 0, alpha=0.5, bottom_margin=4Plots.mm);
  
        # Targets
    @df Data_df[start:start+23,:] scatter!(
        100 .*[:FH_tar :HW_tar], colour =[:firebrick :steelblue], marker=(:circle, stroke(.5)), markersize = 3,
        label=["FH_tar" "HW_tar"], legend=false, legendfontsize=8);

    yaxis!("$(yaxis)", font(10, "serif-roman"), tickfontsize=10)
    xaxis!("time", font(10, "serif-roman"))
end


function bar_actions(Data_df, start, yaxis, title)
    # Battery
    @df Data_df[start:start+23,:] groupedbar(
        [:B :HP], xticks=0:6:24, yticks=-5:2:5, tickfontsize=10, ylim=(-5,5), colour =[:purple :firebrick],
        label=["B" "HP"], legend=false, legendfontsize=8, title="$(title)", titlefontsize=11,
        titlelocation=:left, left_margin=2Plots.mm, linewidth= 0);

    plot!(0:23, ones(24,1)*4.6, linestyle=:dot, linewidth= 2, colour=:purple)
    plot!(0:23, ones(24,1)*-4.6, linestyle=:dot, linewidth= 2, colour=:purple)

    # Heat Pump
   # @df Data_df[start:start+23,:] bar!(
    #    [:HP], colour =:firebrick, label=["HP"]);
    
    plot!(0:23, ones(24,1)*3.0, linestyle=:dot, linewidth= 2, colour=:firebrick)
    plot!(0:23, ones(24,1)*-3.0, linestyle=:dot, linewidth= 2, colour=:firebrick)
    
    plot!(0:23, ones(24,1)*0, linestyle=:dot, linewidth= 3, colour=:black)

    yaxis!("$(yaxis)", font(10, "serif-roman"), tickfontsize=10)
    xaxis!("", font(10, "serif-roman"))
end

function bar_PV(Data_df, start, yaxis, title)
    # PV generation
    @df Data_df[start:start+23,:] plot(
        [:PV_generation], xticks=0:6:24, yticks=0:2:10, tickfontsize=10, ylim=(0,11), colour =:gold,
        label=["g_e"], legend=false, legendfontsize=8, linewidth= 3.0, title="$(title)", titlefontsize=11,
        titlelocation=:left, left_margin=2Plots.mm);

    plot!(0:23, ones(24,1)*10.0, linestyle=:dot, linewidth= 2, colour=:purple)

    # bars PV
    @df Data_df[start:start+23,:] groupedbar!(
        [:PV_HP :PV_GR :PV_B :PV_DE], colour =[:firebrick :grey :purple :orange],
        label=["→ hp" "→ gr" "→ b" "→ de"], legend=false, bar_position = :stack, alpha=0.8);
#=
    # bars B
   @df Data_df[start:start+23,:] bar!(
         (-1) .*[:B_GR], colour =:purple, bar_position = :stack, alpha=0.8);
=#
    # battery cap
    @df Data_df[start:start+23,:] plot!(
        [:Soc_B], colour =:purple, label=["SOC_b"], linewidth= 2.0);

    yaxis!("$(yaxis)", font(10, "serif-roman"), tickfontsize=10)
    xaxis!("", font(10, "serif-roman"))
    annotate!(3, 9, text("$(Data_df[start,:month])/$(Data_df[start,:day])", 10, "serif-roman"))
end

function bar_demand(Data_df, start, yaxis, title)
    # bars
    @df Data_df[start:start+23,:] groupedbar(
        [:B_DE :GR_DE :PV_DE ], colour =[:purple :grey :gold],
        ylim=(0,2.2), yticks=0:0.5:2, xticks=0:6:24, legendfontsize=6,
        label=["B_DE" "GR_DE" "PV_DE"], legend=false, bar_position = :stack, alpha=0.8, title="$(title)",
        titlefontsize=11, titlelocation=:left, left_margin=2Plots.mm);

    # electricity demand
    @df Data_df[start:start+23,:] plot!(
        [:electkwh], colour =:orange,
        label=["d_e"], linewidth= 2.0);

    yaxis!("$(yaxis)", font(10, "serif-roman"), tickfontsize=10)
    xaxis!("", font(10, "serif-roman"))
end

function bar_heat(Data_df, start, yaxis, title)
    # hot water demand
    @df Data_df[start:start+23,:] plot(
        [:hotwaterkwh]+[:heatingkwh], fillrange = [:heatingkwh], yticks=0:2:10, ylim=(0,11), xticks=0:6:24,
        tickfontsize=10, colour =:steelblue, label=["d_fh"], legend=false, alpha=0.2, title="$(title)",
        titlefontsize=11, titlelocation=:left, left_margin=2Plots.mm);

    # heat demand
    @df Data_df[start:start+23,:] plot!(
        [:heatingkwh], fillrange=zeros(24), colour =:firebrick, label=["d_fh"], alpha=0.2);

    # bars
    @df Data_df[start:start+23,:] groupedbar!(
        [:B_HP :GR_HP :PV_HP], colour =[:purple :grey :gold],
        label=["PM_DE" "B_DE" "GR_DE" "PV_DE"], bar_position = :stack, alpha=0.8);

    plot!(0:23, ones(24,1)*3, linestyle=:dot, linewidth= 2, colour = :grey)
    yaxis!("$(yaxis)", font(10, "serif-roman"))
    xaxis!("", font(10, "serif-roman"))
end

function bar_comfort(Data_df, start, yaxis, title)
    @df Data_df[start:start+23,:] groupedbar(
        ([:HP_FH :HP_HW].>0.01)*200, colour =[:firebrick :steelblue],
        ylim=(18,25), yticks=19:1:24, xticks=0:6:24, tickfontsize=10, legendfontsize=6,
        label=["Mod_fh" "Mod_hw"], legend=false, bar_position = :stack, alpha=0.15, title="$(title)", titlefontsize=11,
        titlelocation=:left, left_margin=2Plots.mm);

    # state-of-charge fh
    @df Data_df[start:start+23,:] plot!(
        [:Temp_FH],  colour =:firebrick,
        label=["T_fh"], legend=false, linewidth= 2.0);

    plot!(0:23, ones(24,1)*19, linestyle=:dot, linewidth= 2, colour = :firebrick)
    plot!(0:23, ones(24,1)*24, linestyle=:dot, linewidth= 2, colour = :firebrick)
    yaxis!("$(yaxis)", font(10, "serif-roman"))
    xaxis!("time", font(10, "serif-roman"))
    
    plt = twinx()
    # state-of-charge hw
    @df Data_df[start:start+23,:] plot!(
        plt, [:Vol_HW], ylim=(10,190), yticks=20:40:180, xticks=[], tickfontsize=10, colour = :steelblue,
        label=["V_fh"], legend=false, linewidth= 2.0, linestyle=:dash, grid=false);
    yaxis!(font(10, "serif-roman"))

    plot!(plt, 0:23, ones(24,1)*20, linestyle=:dot, linewidth= 2, colour = :steelblue)
    plot!(plt, 0:23, ones(24,1)*180, linestyle=:dot, linewidth= 2, colour = :steelblue)
end

function bar_row(plotfunction, Data_df, start, yaxis, title, length; legend=true)
    if legend ==true
        plot(plotfunction(Data_df, start, yaxis, title), [plotfunction(Data_df, start+24*(i-1), "", "")
            for i in 2:length]..., plot_legend(plotfunction),layout=grid(1,length+1,
            widths=vcat([((length-.28)/length/length) for i in 1:length], [.28*length/length/length])),
            foreground_color_legend = :transparent, background_color_legend= :transparent);
    else
        plot(plotfunction(Data_df, start, yaxis, title), 
            foreground_color_legend = :transparent, background_color_legend= :transparent);
    end
end

function plot_legend(plotfunction)
    if plotfunction == bar_actions
        plot((1:2)', xlim=(4,8), linestyle=[:solid :solid], colour = [:purple :firebrick], legend=(-.5,.9),
            label=["  B" "  HP"],framestyle= :none, legendfontsize=10,legendfontfamily="serif-roman")
        
    elseif plotfunction == bar_target_B
        scatter((1:1)', xlim=(4,8), colour = :purple, legend=(-.5,.9), alpha=0.5,
            label="  %-SOC_b",framestyle= :none, legendfontsize=10, legendfontfamily="serif-roman",
            marker= (:rect, stroke(0)))
        
        scatter!((1:1)', xlim=(4,8), colour = :purple, legend=(-.5,.9),
            label="  Tar_b",framestyle= :none, marker=(:circle, stroke(1)), legendfontsize=10,legendfontfamily="serif-roman")
        
    elseif plotfunction == bar_target_HP
        scatter((1:2)', xlim=(4,8), colour = [:firebrick :steelblue], legend=(-.5,.9), alpha=0.5,
                    label=["  %-SOC_fh" "  %-SOC_hw"],framestyle= :none, legendfontsize=10,legendfontfamily="serif-roman", 
                    marker=(:rect, stroke(0)))
        
        scatter!((1:2)', xlim=(4,8), colour = [:firebrick :steelblue], legend=(-.5,.9),
                    label=["  Tar_fh" "  Tar_hw"], legendfontsize=10, legendfontfamily="serif-roman",
                    marker= (:circle, stroke(1)))
        
    elseif plotfunction == bar_PV
        scatter((1:4)', xlim=(4,8), colour = [:firebrick :grey :purple :orange], legend=(-.5,.9),
                label=["  pv→hp" "  pv→gr" "  pv→b" "  pv→d_e"], framestyle= :none, marker= (:rect, stroke(0)),
                legendfontsize=10,legendfontfamily="serif-roman")
        plot!((1:3)', xlim=(4,8), linestyle=[:solid :solid :dot], colour = [:gold :purple :purple],
                thickness_scaling=6,label=["  ge" "  Soc_b" "  b_max"])

    elseif plotfunction == bar_demand
        scatter((1:3)', xlim=(4,8), colour = [:grey :purple :gold], legend=(-.5,.5),
            label=["  gr→d_e" "  b→d_e" "  pv→d_e"], framestyle= :none, marker= (:rect, stroke(0)),
            legendfontsize=10,legendfontfamily="serif-roman")
        plot!((1:1)', xlim=(4,8), linestyle=[:solid], colour=:orange, label="  d_e")

    elseif plotfunction == bar_heat
        scatter((1:5)', xlim=(4,8), colour =[:grey :purple :gold :firebrick :steelblue], legend=(-.5,.76),
            label=["  gr→hp" "  b→hp" "  pv→hp" "  d_fh" "  d_hw"], framestyle= :none, marker= (:rect,
                stroke(0)), legendfontsize=10, legendfontfamily="serif-roman", alpha=[1 1 1 .2 .2])
        plot!((1:1)', xlim=(4,8), linestyle=:dot, colour = :grey, label="  hp_max")

    elseif plotfunction == bar_comfort
        scatter((1:2)', xlim=(4,8), colour = [:firebrick :steelblue], legend=(-.3,.64),
            label=["  mod_fh" "  mod_hw"], framestyle= :none, marker= (:rect, stroke(0)),
            legendfontsize=10, legendfontfamily="serif-roman", alpha=[.2 .2])
        plot!((1:4)', xlim=(4,8), linestyle=[:dot :dot :solid :dash],
            colour = [:firebrick :steelblue :firebrick :steelblue], label=["  lim_fh" "  lim_hw" "  T_fh" "  V_hw"])
    end
end

#KPI functions-----------------------------------------------------------------------------------------
function calc_profit(Data_df; ratio=3)
    return convert.(Float64, 
               round.(sum(Data_df[!,"p_sell"] .* Data_df[!,"PV_GR"] - 
                    (ratio * Data_df[!,"p_sell"]) .* sum(Data_df[!,j] for j in ["GR_DE", "GR_HP"])), digits=2))
end

function calc_energy_use(Data_df)
    return convert.(Float64, 
                round.(sum(
                    sum(Data_df[!,j] for j in ["PV_DE", "B_DE", "GR_DE", "HP_FH", "HP_HW"])), digits=2))
end

function calc_self_consumption(Data_df)
    gd = groupby(Data_df, ["month", "day"])
    Data_df_sum = combine(gd, valuecols(gd) .=> sum, renamecols=false)
    Data_df_sum[!,"SeCo"] = ((Data_df_sum[!,"PV_DE"] .+
                                Data_df_sum[!,"PV_HP"] .+
                                    Data_df_sum[!,"PV_B"]) ./ Data_df_sum[!,"PV_generation"])

    return convert.(Float64, 
               round.(100 * sum(Data_df_sum[:,"SeCo"]) / nrow(Data_df_sum), digits=2))
end

function calc_self_sufficiency(Data_df)
    gd = groupby(Data_df, ["month", "day"])
    Data_df_sum = combine(gd, valuecols(gd) .=> sum, renamecols=false)
    Data_df_sum[!,"SeSu"] = 1 .- (Data_df_sum[!,"GR_DE"] .+ Data_df_sum[!,"GR_HP"]) ./ 
                    sum(Data_df_sum[!,j] for j in ["PV_DE", "B_DE", "GR_DE", "HP_FH", "HP_HW"])
    return convert.(Float64, 
               round.(100 * sum(Data_df_sum[!,"SeSu"]) / nrow(Data_df_sum), digits=2))
end

function calc_comfort_violations(Data_df)
    return convert.(Float64, 
        round.(sum(sum(Data_df[!,j] for j in ["V_HW_plus", "V_HW_minus", "T_FH_plus", "T_FH_minus"])), digits=2))
end
