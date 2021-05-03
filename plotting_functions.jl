function bar_PV(Data_df, date, yaxis, title)
    # PV generation
    @df Data_df[(Data_df[:,:day].==Dates.day(date)) .&(Data_df[:,:month].==Dates.month(date)),:] plot(
        [:PV_generation], xticks=0:6:24, yticks=-2:2:10, tickfontsize=10, ylim=(-2,11), color =[:gold],
        label=["g_e"], legend=false, legendfontsize=8, linewidth= 3.0, title="$(title)", titlefontsize=11,
        titlelocation=:left);

    plot!(0:23, ones(24,1)*10.0, linestyle=:dot, linewidth= 2, color=:purple)

    # bars PV
    @df Data_df[(Data_df[:,:day].==Dates.day(date)) .&(Data_df[:,:month].==Dates.month(date)),:] groupedbar!(
        [:PV_HP :PV_GR :PV_B :PV_DE], color =[:firebrick :grey :purple :orange],
        label=["→ hp" "→ gr" "→ b" "→ de"], legend=false, bar_position = :stack, alpha=0.8);

    # bars B
    @df Data_df[(Data_df[:,:day].==Dates.day(date)) .&(Data_df[:,:month].==Dates.month(date)),:] groupedbar!(
        (-1) .*[:B_HP :B_DE], color =[:firebrick :orange],
        label=[], bar_position = :stack, alpha=0.8);

    # battery cap
    @df Data_df[(Data_df[:,:day].==Dates.day(date)) .&(Data_df[:,:month].==Dates.month(date)),:] plot!(
        [:Soc_B], color =[:purple], label=["SOC_b"], linewidth= 2.0);

    yaxis!("$(yaxis)", font(10, "serif"), tickfontsize=10)
    annotate!(3, 8.5, text("$(Dates.month(date))/$(Dates.day(date))", 10))
end

function bar_demand(Data_df, date, yaxis, title)
    # bars
    @df Data_df[(Data_df[:,:day].==Dates.day(date)) .&(Data_df[:,:month].==Dates.month(date)),:] groupedbar(
        [:B_DE :GR_DE :PV_DE ], color =[:purple :grey :gold],
        ylim=(0,2.2), yticks=0:0.5:2, xticks=0:6:24, legendfontsize=6,
        label=["B_DE" "GR_DE" "PV_DE"], legend=false, bar_position = :stack, alpha=0.8, title="$(title)",
        titlefontsize=11, titlelocation=:left);

    # electricity demand
    @df Data_df[(Data_df[:,:day].==Dates.day(date)) .&(Data_df[:,:month].==Dates.month(date)),:] plot!(
        [:electkwh], color =[:orange],
        label=["d_e"], linewidth= 2.0);

    yaxis!("$(yaxis)", font(10, "serif"), tickfontsize=10)
end

function bar_heat(Data_df, date, yaxis, title)
    # hot water demand
    @df Data_df[(Data_df[:,:day].==Dates.day(date)) .&(Data_df[:,:month].==Dates.month(date)),:] plot(
        [:hotwaterkwh]+[:heatingkwh], fillrange=[:heatingkwh], yticks=0:2:10, ylim=(0,11), xticks=0:6:24,
        tickfontsize=10, color =[:steelblue], label=["d_fh"], legend=false, alpha=0.2, title="$(title)",
        titlefontsize=11, titlelocation=:left);

    # heat demand
    @df Data_df[(Data_df[:,:day].==Dates.day(date)) .&(Data_df[:,:month].==Dates.month(date)),:] plot!(
        [:heatingkwh], fillrange=zeros(24), color =[:firebrick], label=["d_fh"], alpha=0.2);

    # bars
    @df Data_df[(Data_df[:,:day].==Dates.day(date)) .&(Data_df[:,:month].==Dates.month(date)),:] groupedbar!(
        [:B_HP :GR_HP :PV_HP], color =[:purple :grey :gold],
        label=["PM_DE" "B_DE" "GR_DE" "PV_DE"], bar_position = :stack, alpha=0.8);

    plot!(0:23, ones(24,1)*3, linestyle=:dot, linewidth= 2, color=:grey)
    yaxis!("$(yaxis)", font(10, "serif"))
end

function bar_comfort(Data_df, date, yaxis, title)
    @df Data_df[(Data_df[:,:day].==Dates.day(date)) .&(Data_df[:,:month].==Dates.month(date)),:] groupedbar(
        ([:HP_FH :HP_HW].>0.005)*200, color =[:firebrick :steelblue],
        ylim=(19,23), yticks=20:1:22, xticks=0:6:24, tickfontsize=10, legendfontsize=6,
        label=["Mod_fh" "Mod_hw"], legend=false, bar_position = :stack, alpha=0.15, title="$(title)", titlefontsize=11,
        titlelocation=:left);

    # state-of-charge fh
    @df Data_df[(Data_df[:,:day].==Dates.day(date)) .&(Data_df[:,:month].==Dates.month(date)),:] plot!(
        [:Temp_FH],  color =[:firebrick],
        label=["T_fh"], legend=false, linewidth= 2.0);

    plot!(0:23, ones(24,1)*20, linestyle=:dot, linewidth= 2, color=:firebrick)
    plot!(0:23, ones(24,1)*22, linestyle=:dot, linewidth= 2, color=:firebrick)
    yaxis!("$(yaxis)", font(10, "serif"))
    xaxis!("time")

    plt = twinx()
    # state-of-charge hw
    @df Data_df[(Data_df[:,:day].==Dates.day(date)) .&(Data_df[:,:month].==Dates.month(date)),:] plot!(
        plt, [:Vol_HW], ylim=(10,190), yticks=20:40:180, xticks=[], tickfontsize=10, color =[:steelblue],
        label=["V_fh"], legend=false, linewidth= 2.0, linestyle=:dash, grid=false);

    plot!(plt, 0:23, ones(24,1)*20, linestyle=:dot, linewidth= 2, color=:steelblue)
    plot!(plt, 0:23, ones(24,1)*180, linestyle=:dot, linewidth= 2, color=:steelblue)
end

function bar_row(plotfunction, Data_df, date, yaxis, title, length)
        plot(plotfunction(Data_df, date, yaxis, title), [plotfunction(Data_df, (date+Day(i-1)), "", "")
            for i in 2:length]..., plot_legend(plotfunction),layout=grid(1,length+1,
            widths=vcat([((length-.28)/length/length) for i in 1:length], [.28*length/length/length])),
            size=(300*(length+1),200), foreground_color_legend = :transparent, background_color_legend= :transparent);
end

function plot_legend(plotfunction)
    if plotfunction == bar_PV
        scatter((1:4)', xlim=(4,8), color =[:firebrick :grey :purple :orange], legend=(-.5,.9),
            label=["  →hp" "  →gr" "  →b" "  →d_e"], framestyle= :none, marker= (:rect, stroke(0)),
            legendfontsize=10)
        plot!((1:3)', xlim=(4,8), linestyle=[:solid :solid :dot], color=[:gold :purple :purple],
            label=["  ge" "  SOC_b" "  b_max"])

        elseif plotfunction == bar_demand
        scatter((1:3)', xlim=(4,8), color =[:grey :purple :gold], legend=(-.5,.5),
            label=["  gr→d_e" "  b→d_e" "  pv→d_e"], framestyle= :none, marker= (:rect, stroke(0)),
            legendfontsize=10)
        plot!((1:1)', xlim=(4,8), linestyle=[:solid], color=:orange, label="  de")

        elseif plotfunction == bar_heat
        scatter((1:5)', xlim=(4,8), color =[:grey :purple :gold :firebrick :steelblue], legend=(-.5,.76),
            label=["  gr→hp" "  b→hp" "  pv→hp" "  d_fh" "  d_hw"], framestyle= :none, marker= (:rect,
                stroke(0)),
            legendfontsize=10, alpha=[1 1 1 1 .2 .2])
        plot!((1:1)', xlim=(4,8), linestyle=[:dot], color=:grey, label="  hp_max")

        elseif plotfunction == bar_comfort
        scatter((1:2)', xlim=(4,8), color =[:firebrick :steelblue], legend=(-.3,.64),
            label=["  mod_fh" "  mod_hw"], framestyle= :none, marker= (:rect, stroke(0)),
            legendfontsize=10, alpha=[.2 .2])
        plot!((1:4)', xlim=(4,8), linestyle=[:dot :dot :solid :dash],
            color=[:firebrick :steelblue :firebrick :steelblue], label=["  range_fh" "  range_hw" "  T_fh" "  V_hw"])
    end
end

#KPI functions-----------------------------------------------------------------------------------------
function calc_profit(Data_df)
    return convert.(Float64, round.(sum(Data_df[!,"profits"]), digits=2))
end

function calc_energy_use(Data_df)
    return convert.(Int64, round.(sum(sum(Data_df[!,j] for j in ["PV_DE", "B_DE", "GR_DE", "HP_FH", "HP_HW"]))))
end

function calc_self_consumption(Data_df)
    return convert.(Int64, round.(100 * sum((1 .- (Data_df[!,"PV_GR"] / sum(Data_df[!,"PV_generation"]))))/nrow(Data_df), digits=0))
end

function calc_self_sufficiency(Data_df)
    return convert.(Int64, round.(100 * sum((1 .- (sum(Data_df[!,j] for j in ["GR_DE", "GR_HP"]) ./ calc_energy_use(Data_df))))/nrow(Data_df), digits=0))
end

function calc_comfort_violations(Data_df)
    return convert.(Int64, round.(sum(sum(Data_df[!,j] for j in ["V_HW_plus", "V_HW_minus", "T_FH_plus", "T_FH_minus"])), digits=0))
end
