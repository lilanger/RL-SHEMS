include("main.jl");
include("SHEMS_optimizer.jl");

function yearly_SHEMS(h_start=1, h_end=8760, season="summer", costfactor=1.0, outputflag=0, case="eval")

    # Initialize technical setup according to case______________________
    # set_SHEMS_parameters(h_start, h_end, h_predict, h_control, rolling_flag, costfactor)
    sh, hp, fh, hw, b, m = set_SHEMS_parameters(h_start, h_end, (h_end-h_start)+1, (h_end-h_start)+1,
        false, costfactor, outputflag);

    results  = SHEMS_optimizer(sh, hp, fh, hw, b, m, season, case);

    # write to results folder___________________________________________________
    write_to_results_file(hcat(results, ones(size(results,1))*m.h_predict), m, season, costfactor, case)
    return nothing
end

function set_SHEMS_parameters(h_start, h_end, h_predict, h_control, rolling_flag, costfactor=1.0, outputflag=0)

    # Initialize technical setup________________________________________________
    # Model_SHEMS(h_start, h_end, h_predict, h_control, big, rolling_flag, solver, mip_gap, output_flag, presolve_flag)
    m = Model_SHEMS(h_start, h_end,  h_predict, h_control, 60, rolling_flag, "Cbc", 0.005f0, outputflag, -1);
    # HeatPump(eta, rate_max)
    hp = HeatPump(1.0f0, 3.0f0);
    # ThermalStorage(eta, volume, loss, t_supply, soc_min, soc_max)
    fh = ThermalStorage(1.0f0, 10.0f0, 0.045f0, 30.0f0, 19.0f0, 24.0f0);
    hw = ThermalStorage(1.0f0, 200.0f0, 0.035f0, 45.0f0, 20.0f0, 180.0f0);

    # Battery(eta, soc_min, soc_max, rate_max, loss)
    b = Battery(0.98f0, 0.0f0, 10.0f0, 4.6f0, 0.00003f0);
    # SHEMS(costfactor, p_buy, p_sell, soc_b, soc_fh, soc_hw, h_start)
    sh = SHEMS(costfactor, 0.3f0, 0.1f0, 0.5 * (b.soc_min + b.soc_max),
                                         0.5 * (fh.soc_min + fh.soc_max),
                                         0.5 * (hw.soc_min + hw.soc_max), h_start);
    return sh, hp, fh, hw, b, m
end

function write_to_results_file(results, m, season="summer", costfactor=1.0, case="eval")
    date=210705;
    CSV.write("benchmark_opt/results/$(date)_results_$(case)_"*
        "$(season)_$(costfactor).csv", DataFrame(results, :auto), header=["Temp_FH", "Vol_HW",
            "Soc_B", "V_HW_plus", "V_HW_minus", "T_FH_plus", "T_FH_minus", "profits", "comfort", "COP_FH",
            "COP_HW","PV_DE", "B_DE", "GR_DE", "PV_B", "PV_GR", "PV_HP","GR_HP", "B_HP", "HP_FH", "HP_HW",
            "month", "day", "hour", "horizon"]);
    return nothing
end

function COPcalc(ts, t_outside)
    # Calculate coefficients of performance for every time period (1:h_predict)
    return cop = max.((5.8*ones(size(t_outside,1))) -(1. /14) * abs.((ts.t_supply*ones(size(t_outside,1)))
            -t_outside), 0);
end

# also change path to input file in optimizer
########### EVALUATIOM #######################################
# yearly_SHEMS(1, 360, "summer", 1.0, 0, "eval") #solution time 3 s
# yearly_SHEMS(1, 360, "winter", 1.0, 0, "eval") #solution time 3902 s
#yearly_SHEMS(1, 1440, "all", 1.0, 0, "eval") #solution time 6775 s
# yearly_SHEMS(1, 720, "both", 1.0, 0, "eval") #solution time

########### TESTING #######################################
# yearly_SHEMS(1, 768, "summer", 1.0, 0, "test") #solution time 696 s
# yearly_SHEMS(1, 720, "winter", 1.0, 0, "test") #solution time 88 s
 yearly_SHEMS(1, 3000, "all", 1.0, 0, "test") #solution time
# yearly_SHEMS(1, 1488, "both", 1.0, 0, "test") #solution time
