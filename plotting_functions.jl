function read_data(season; network="Opt", stop="final")
    date = "2021-05-29"
    version = "v12"
    NUM_STEPS = 24
    NUM_EP = 3001
    
    if stop == "final"
        idx=NUM_EP
    elseif stop == "best"
        idx="best"
    end
    
   if network == "Yu"
        L1 = 300
        L2 = 600
   elseif network == "Pendulum"
       L1 = 400
       L2 = 300   
    elseif network == "Opt"
        if season == "all"
            Flow_df = CSV.read("benchmark_opt/results/210521_results_1440_1440_1-1440_all_1.0.csv", DataFrame);
        else
            Flow_df = CSV.read("benchmark_opt/results/210521_results_360_360_1-360_$(season)_1.0.csv", DataFrame);
        end
        Input_df = CSV.read("data/$(season)_evaluation.csv", DataFrame);
        Data_df = hcat(Flow_df[1:end-1, 1:end-4], Input_df[1:end-1, :]) # cut index + hour + last input (no action)
        return Data_df
    end
    
    #case = "$(season)_no-L2_buffer-12T"
    #if case == "24T"
      #  case = "$(season)_L2_no-noise-scale_24T-120"
   # elseif case =="50T"
    case = "$(season)_L2_no-noise-scale"
    #case = "$(season)_L2_nns_term-off+pain=200"
   # end
    
    
    Input_df = CSV.read("data/$(season)_evaluation.csv", DataFrame);
    Flow_df = CSV.read("out/$(date)_results_$(version)_$(NUM_STEPS)_$(NUM_EP)_$(L1)_$(L2)_$(case)_$(idx).csv", DataFrame);
    Data_df = hcat(Flow_df[1:end, :], Input_df[1:end-1, :]) # cut index + hour + last input (no action)
    return Data_df
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

    yaxis!("$(yaxis)", font(10, "serif"), tickfontsize=10)
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

    # bars B
   # @df Data_df[start:start+23,:] groupedbar!(
     #   (-1) .*[:B_HP :B_DE], colour =[:firebrick :orange],
    #    label=[], bar_position = :stack, alpha=0.8);

    # battery cap
    @df Data_df[start:start+23,:] plot!(
        [:Soc_B], colour =:purple, label=["SOC_b"], linewidth= 2.0);

    yaxis!("$(yaxis)", font(10, "serif"), tickfontsize=10)
    annotate!(3, 8.5, text("$(Data_df[start,:month])/$(Data_df[start,:day])", 10))
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

    yaxis!("$(yaxis)", font(10, "serif"), tickfontsize=10)
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
    yaxis!("$(yaxis)", font(10, "serif"))
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
    yaxis!("$(yaxis)", font(10, "serif"))
    xaxis!("time")

    plt = twinx()
    # state-of-charge hw
    @df Data_df[start:start+23,:] plot!(
        plt, [:Vol_HW], ylim=(10,190), yticks=20:40:180, xticks=[], tickfontsize=10, colour = :steelblue,
        label=["V_fh"], legend=false, linewidth= 2.0, linestyle=:dash, grid=false);

    plot!(plt, 0:23, ones(24,1)*20, linestyle=:dot, linewidth= 2, colour = :steelblue)
    plot!(plt, 0:23, ones(24,1)*180, linestyle=:dot, linewidth= 2, colour = :steelblue)
end

function bar_row(plotfunction, Data_df, start, yaxis, title, length)
        plot(plotfunction(Data_df, start, yaxis, title), [plotfunction(Data_df, start+24*(i-1), "", "")
            for i in 2:length]..., plot_legend(plotfunction),layout=grid(1,length+1,
            widths=vcat([((length-.28)/length/length) for i in 1:length], [.28*length/length/length])),
            size=(300*(length+1),220), foreground_color_legend = :transparent, background_color_legend= :transparent);
end

function plot_legend(plotfunction)
    if plotfunction == bar_actions
        plot((1:2)', xlim=(4,8), linestyle=[:solid :solid], colour = [:purple :firebrick], legend=(-.5,.9),
            label=["  B" "  HP"],framestyle= :none, legendfontsize=10)
        
    elseif plotfunction == bar_PV
        scatter((1:4)', xlim=(4,8), colour = [:firebrick :grey :purple :orange], legend=(-.5,.9),
            label=["  →hp" "  →gr" "  →b" "  →d_e"], framestyle= :none, marker= (:rect, stroke(0)),
            legendfontsize=10)
        plot!((1:3)', xlim=(4,8), linestyle=[:solid :solid :dot], colour = [:gold :purple :purple],
            label=["  ge" "  SOC_b" "  b_max"], linewidth=3)

    elseif plotfunction == bar_demand
        scatter((1:3)', xlim=(4,8), colour = [:grey :purple :gold], legend=(-.5,.5),
            label=["  gr→d_e" "  b→d_e" "  pv→d_e"], framestyle= :none, marker= (:rect, stroke(0)),
            legendfontsize=10)
        plot!((1:1)', xlim=(4,8), linestyle=[:solid], colour=:orange, label="  de")

    elseif plotfunction == bar_heat
        scatter((1:5)', xlim=(4,8), colour =[:grey :purple :gold :firebrick :steelblue], legend=(-.5,.76),
            label=["  gr→hp" "  b→hp" "  pv→hp" "  d_fh" "  d_hw"], framestyle= :none, marker= (:rect,
                stroke(0)),
            legendfontsize=10, alpha=[1 1 1 1 .2 .2])
        plot!((1:1)', xlim=(4,8), linestyle=:dot, colour = :grey, label="  hp_max")

    elseif plotfunction == bar_comfort
        scatter((1:2)', xlim=(4,8), colour = [:firebrick :steelblue], legend=(-.3,.64),
            label=["  mod_fh" "  mod_hw"], framestyle= :none, marker= (:rect, stroke(0)),
            legendfontsize=10, alpha=[.2 .2])
        plot!((1:4)', xlim=(4,8), linestyle=[:dot :dot :solid :dash],
            colour = [:firebrick :steelblue :firebrick :steelblue], label=["  lim_fh" "  lim_hw" "  T_fh" "  V_hw"])
    end
end

#KPI functions-----------------------------------------------------------------------------------------
function calc_profit(Data_df; p_sell=0.1)
    return convert.(Float64, 
               round.(sum(p_sell * Data_df[!,"PV_GR"] - 
                    0.3 * sum(Data_df[!,j] for j in ["GR_DE", "GR_HP"])), digits=2))
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
