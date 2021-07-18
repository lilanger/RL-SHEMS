module ShemsEnv
# Ported from: https://github.com/openai/gym/blob/996e5115621bf57b34b9b79941e629a36a709ea1/gym/envs/classic_control/pendulum.py
#              https://github.com/openai/gym/wiki/Pendulum-v0
# add DataFrames to dependencies
# add shems environment to Reinforce import

using Reinforce: AbstractEnvironment
using LearnBase: IntervalSet
using RecipesBase
using Distributions: Uniform
using Random
using DataFrames, CSV

import Reinforce: reset!, actions, action, finished, step!, state

export
  Shems,
  reset!,
  step!,
  actions,
  action,
  finished,
  state,
  track

struct HeatPump
    rate_max::Float32
end

struct PV
    eta::Float32
end

struct Battery
    eta::Float32
    soc_min::Float32
    soc_max::Float32
    rate_max::Float64
    loss::Float32
end

struct ThermalStorage
    volume::Float32
    loss::Float32
    t_supply::Float32
    soc_min::Float32
    soc_max::Float32
end

struct Market
    p_buy::Float64
	p_sell::Float64
	comfort_weight::Float64
end

# PV(eta)
pv = PV(0.95f0);
# HeatPump(rate_max)
hp = HeatPump(3f0);
# ThermalStorage(volume, loss, t_supply, soc_min, soc_max)
fh = ThermalStorage(10f0, 0.045f0, 30f0, 19f0, 24f0); ##YU
hw = ThermalStorage(200f0, 0.035f0, 45f0, 20f0, 180f0);
# Battery(eta, soc_min, soc_max, rate_max, loss)
b = Battery(0.98f0, 0f0, 10f0, 4.6f0, 0.00003f0);
# Market(price, comfort_weight)
m = Market(0.3f0, 0.1f0, 1f0)

const p_concr = 2400.0f0;   # kg/m^3
const c_concr = 1f0;      # kJ/(kg*°C)
const p_water = 997f0;    # kg/m^3
const c_water = 4.184f0;    # kJ/(kg*°C)

mutable struct ShemsState{T<:AbstractFloat} <: AbstractVector{T}
  Soc_b::T
  T_fh::T
  V_hw::T

  d_e::T
  d_fh::T
  d_hw::T

  g_e::T
  t_out::T
  h::T
  # h_cos::T
  # h_sin::T
end

ShemsState() = ShemsState(0f0, 22f0, 180f0, 0f0, 0f0, 0f0, 0f0, 0f0, 0f0)#1f0, 0f0)

Base.size(::ShemsState) = (9,)
# Base.size(::ShemsState) = (10,)

function Base.getindex(s::ShemsState, i::Int)
  (i > length(s)) && throw(BoundsError(s, i))
  ifelse(i == 1, s.Soc_b,
  	ifelse(i == 2, s.T_fh,
		ifelse(i == 3, s.V_hw,
			ifelse(i == 4, s.d_e,
				ifelse(i == 5, s.d_fh,
					ifelse(i == 6, s.d_hw,
						ifelse(i == 7, s.g_e,
							ifelse(i == 8, s.t_out,
								s.h))))))))
								# ifelse(i == 9, s.h_cos,
								# 		s.h_sin)))))))))
end

function Base.setindex!(s::ShemsState, x, i::Int)
  (i > length(s)) && throw(BoundsError(s, i))
  setproperty!(s, ifelse(i == 1, :Soc_b,
				  	ifelse(i == 2, :T_fh,
				  		ifelse(i == 3, :V_hw,
				  			ifelse(i == 4, :d_e,
				  				ifelse(i == 5, :d_fh,
				  					ifelse(i == 6, :d_hw,
				  						ifelse(i == 7, :g_e,
				  							ifelse(i == 8, :t_out,
												:h)))))))), x)
												# ifelse(i == 9, :h_cos,
				  								# 		:h_sin))))))))), x)
	end

mutable struct ShemsAction{T<:AbstractFloat} <: AbstractVector{T}
  B_switch::T
  B::T
  HP_switch::T
  FH::T
  HW::T
end

ShemsAction() = ShemsAction(0f0, 0f0, 0f0, 0f0, 0f0)

Base.size(::ShemsAction) = (5,)

function Base.getindex(a::ShemsAction, i::Int)
  (i > length(a)) && throw(BoundsError(a, i))
  ifelse(i == 1, a.B_switch,
  	ifelse(i == 2, a.B,
		ifelse(i == 3, a.HP_switch,
			ifelse(i == 4, a.FH,
  				a.HW))))
end

function Base.setindex!(a::ShemsAction, x, i::Int)
  (i > length(a)) && throw(BoundsError(a, i))
  setproperty!(a, ifelse(i == 1, :B_switch,
  					ifelse(i == 2, :B,
						ifelse(i == 3, :HP_switch,
							ifelse(i == 4, :FH,
  								:HW)))), x)
end

mutable struct Shems{V<:AbstractVector, W<:AbstractVector} <: AbstractEnvironment
  state::V
  reward::Float64
  a::W
  step::Int
  maxsteps::Int
  idx::Int
  path::String
end

Base.size(::Shems) = (7,)

function Base.getindex(env::Shems, i::Int)
  (i > length(env)) && throw(BoundsError(env, i))
  ifelse(i == 1, env.state,
  	ifelse(i == 2, env.reward,
		ifelse(i == 3, env.a,
			ifelse(i == 4, env.step,
				ifelse(i == 5, env.maxsteps,
					ifelse(i == 6, env.idx,
						env.path))))))
end

function Base.setindex!(env::Shems, x, i::Int)
  (i > length(env)) && throw(BoundsError(env, i))
  setproperty!(env, ifelse(i == 1, :state,
	  				  	ifelse(i == 2, :reward,
					  		ifelse(i == 3, :a,
					  			ifelse(i == 4, :step,
					  				ifelse(i == 5, :maxsteps,
										ifelse(i == 6, :idx,
					  						:path)))))), x)
end

Shems(maxsteps, path) = Shems(ShemsState(), 0.0, ShemsAction(), 0, maxsteps, 1, path)

function COPcalc(ts::ThermalStorage, env::Shems)
    # Calculate coefficients of performance for time period
    return max(5.8 -(1/14 * abs(ts.t_supply - env.state.t_out)), 0);
end

function IsHot(env::Shems)
    # Determine if outside temperture is hotter than inside temperture
    return env.state.t_out > env.state.T_fh
end

function reset!(env::Shems; rng=0)
  idx = reset_state!(env; rng=rng)
  env.reward = 0.0
  env.a = ShemsAction()
  env.step = 0
  env.idx = idx
  env.path = env.path
  return env
end

function reset_state!(env::Shems; rng=0)
	if rng == 0 #training
		env.state.Soc_b = rand(Uniform(b.soc_min, b.soc_max))
		env.state.T_fh = rand(Uniform(fh.soc_min, fh.soc_max))
		env.state.V_hw = rand(Uniform(hw.soc_min, hw.soc_max))
	elseif rng == -1 #tracking
		env.state.Soc_b = 0.5 * (b.soc_min + b.soc_max)
		env.state.T_fh = 0.5 * (fh.soc_min + fh.soc_max)
		env.state.V_hw = 0.5 * (hw.soc_min + hw.soc_max)
    else #evaluation/testing
		env.state.Soc_b = rand(MersenneTwister(rng), Uniform(b.soc_min, b.soc_max))
		env.state.T_fh = rand(MersenneTwister(rng), Uniform(fh.soc_min, fh.soc_max))
		env.state.V_hw = rand(MersenneTwister(rng), Uniform(hw.soc_min, hw.soc_max))
	end

	df = CSV.read(env.path, DataFrame)
	idx = rand(1:(nrow(df) - env.maxsteps))
	env.state.d_e = df[idx, :electkwh]
	env.state.d_fh = df[idx,:heatingkwh]
    env.state.d_hw = df[idx,:hotwaterkwh]
	env.state.g_e = df[idx,:PV_generation]
	env.state.t_out = df[idx,:Temperature]
	env.state.h = df[idx,:hour]
	# env.state.h_cos = df[idx,:hour_cos]
	# env.state.h_sin = df[idx,:hour_sin]
	return idx
end

function next_state!(env::Shems)
	df = CSV.read(env.path, DataFrame)
	idx = env.idx + 1

	env.state.d_e = df[idx, :electkwh]
	env.state.d_fh = df[idx,:heatingkwh]
    env.state.d_hw = df[idx,:hotwaterkwh]
	env.state.g_e = df[idx,:PV_generation]
	env.state.t_out = df[idx,:Temperature]
	env.state.h = df[idx,:hour]
	# env.state.h_cos = df[idx,:hour_cos]
	# env.state.h_sin = df[idx,:hour_sin]
	return nothing
end

actions(env::Shems, s) = IntervalSet([0f0, 0f0, 0f0, 0f0, 0f0], [1f0, b.rate_max, 1f0, hp.rate_max, hp.rate_max])
#actions(env::Shems, s) = IntervalSet([0f0, -1f0], [1f0, 1f0])

function action(env::Shems, track=-1)
	if track == -1
		Soc_b, T_fh, V_hw, d_e, d_fh, d_hw, g_e, t_out, h = env.state #h_cos, h_sin = env.state
		################### Heat pump ########################################
		# HP percentage SOCs
		T_fh_perc = (T_fh - fh.soc_min) / (fh.soc_max - fh.soc_min)
		V_hw_perc = (V_hw - hw.soc_min) / (hw.soc_max - hw.soc_min)

		# charge HP when under threshold, choose emptiest
		if T_fh_perc < 0.7 || V_hw_perc < 0.7 # under threshold 70%
			if T_fh_perc <= V_hw_perc # fully charge FH
				cop_fh = COPcalc(fh, env)
				Hot = IsHot(env)
				HP_switch = 1
				FH = 1/cop_fh * ( (((p_concr * fh.volume * c_concr) * (fh.soc_max - T_fh)) / (60 * 60))+
						d_fh +( (1 - Hot) * fh.loss - Hot * fh.loss ) - 1f-6)
				FH = min(hp.rate_max, FH)
			elseif V_hw_perc < T_fh_perc # fully charge HW
				cop_hw = COPcalc(hw, env)
				HP_switch = 0
				HW = 1/cop_hw * ( ((((p_water * hw.t_supply * c_water)  / 1000)* (hw.soc_max - V_hw)) / (60 * 60)) +
						d_hw + hw.loss - 1f-6)
				HW = min(hp.rate_max, HW)
			end
		else
			HP = 0
		end
		############################# Battery ###############################
		# charge battery when PV is available (substracting electr. demand)
		pv_ = g_e - d_e

		# charge to max if SOC less than 95%
		if pv_ > 0 && Soc_b < (0.95 * b.soc_max)
			B_switch = 1
			B = clamp(pv_, 0, min(b.rate_max, b.soc_max - Soc_b) + b.loss)
		# discharge at max if no PV available (level regulated in step!)
		elseif pv_ <= 0 && Soc_b > 1f-4
			B_switch = 0
			B = min(b.rate_max,  ((1 - b.loss) *Soc_b))
		else
			B = 0
		end
	elseif track == -2
		Soc_b, T_fh, V_hw, d_e, d_fh, d_hw, g_e, t_out, h = env.state #h_cos, h_sin = env.state
		################### Heat pump ########################################
		# HP percentage SOCs
		T_fh_perc = (T_fh - fh.soc_min) / (fh.soc_max - fh.soc_min)
		V_hw_perc = (V_hw - hw.soc_min) / (hw.soc_max - hw.soc_min)

		# charge HP when under threshold, choose emptiest
		if T_fh_perc < 0.7 || V_hw_perc < 0.7 # under threshold 70%
			if T_fh_perc <= V_hw_perc # fully charge FH
				cop_fh = COPcalc(fh, env)
				Hot = IsHot(env)
				HP_switch = 1
				FH = 1/cop_fh * ( (((p_concr * fh.volume * c_concr) * (fh.soc_max - T_fh)) / (60 * 60))) #+
						#d_fh +( (1 - Hot) * fh.loss - Hot * fh.loss ) - 1f-6)
				FH = min(hp.rate_max, FH)
			elseif V_hw_perc < T_fh_perc # fully charge HW
				cop_hw = COPcalc(hw, env)
				HP_switch = 0
				HW = 1/cop_hw * ( ((((p_water * hw.t_supply * c_water)  / 1000)* (hw.soc_max - V_hw)) / (60 * 60))) #+
						#d_hw + hw.loss - 1f-6)
				HW = min(hp.rate_max, HW)
			end
		else
			FH = 0
			HW = 0
		end
		############################# Battery ###############################
		# charge battery when PV is available (substracting electr. demand)
		pv_ = g_e - d_e

		# charge to max if SOC less than 95%
		if pv_ > 0 && Soc_b < (0.95 * b.soc_max)
			B_switch = 1
			B = clamp(pv_, 0, min(b.rate_max, b.soc_max - Soc_b))# + b.loss) )
		# discharge at max if no PV available (level regulated in step!)
		elseif pv_ <= 0 && Soc_b > 1f-4
			B_switch = 0
			B = min(b.rate_max,  ((1 - b.loss) *Soc_b))
		else
			B = 0
		end
	end

	return Float32.([B_switch, B, HP_switch, FH, HW])
end

function step!(env::Shems, s, a; track=0)
	Soc_b, T_fh, V_hw, d_e, d_fh, d_hw, g_e, t_out, h = env.state #h_cos, h_sin = env.state
	B_switch, B, HP_switch, FH, HW = a
	env.a = ShemsAction(B_switch, B, HP_switch, FH, HW)
  	pv_, BD, BC, T_fh_plus, T_fh_minus, V_hw_plus, V_hw_minus, cop_fh, cop_hw, abort, comfort = zeros(11)
	PV_DE, PV_B, PV_HP, PV_GR, B_DE, B_HP, GR_DE, GR_HP, HP_FH, HP_HW = zeros(10)

	############# DETERMINE FLOWS ###################################
	if B_switch < 0.5 && B > 0.01 # battery discharging, restrictions discharging rate and soc
		BD = clamp(B, 0, min(b.rate_max,  ((1 - b.loss - 1f-7) *Soc_b)))
	end

	if HP_switch > 0.5 && FH > 0.01 # floor heating
		HP_FH = clamp(FH, 0, hp.rate_max)
	elseif HP_switch < 0.5 && HW > 0.01 # hot water
		HP_HW = clamp(HW, 0, hp.rate_max)
	end

	#---------------- PV generation greater than electricity demand -------------
	if (g_e * pv.eta) > d_e
		PV_DE = d_e
  		pv_ = (g_e * pv.eta) - PV_DE # PV left
		# heat pump (only HP_FH or HP_HW can be >0)
		if  pv_ > (HP_FH + HP_HW)
			PV_HP = (HP_FH + HP_HW)
			pv_ -= PV_HP
		elseif pv_ <= (HP_FH + HP_HW)
			PV_HP = pv_
			pv_ = 0
			if BD > (HP_FH + HP_HW - PV_HP) / b.eta # heat pump from battery?
				B_HP = (HP_FH + HP_HW - PV_HP)
				BD -= B_HP / b.eta
			elseif BD <= (HP_FH + HP_HW - PV_HP) / b.eta
				B_HP = BD * b.eta
				GR_HP = (HP_FH + HP_HW - PV_HP) - B_HP 		# slack variable heat pump
			end
		end

		# battery charging (only from PV, has to be adapted for dynamic prices!)
		if B_switch > 0.5 && B > 0.01
			PV_B = clamp(B, 0, min(pv_, b.rate_max, b.soc_max - Soc_b) )
			pv_ -= PV_B
		end
		PV_GR = pv_ 								# slack variable PV generation

	# -------------- not enough PV for electr. demand --------------------------
	elseif (g_e * pv.eta) <= d_e # electr. demand
		PV_DE = g_e * pv.eta
		d_e -= PV_DE
		if BD > d_e / b.eta # from battery?
			B_DE = d_e
			BD -= B_DE / b.eta
			if BD > (HP_FH + HP_HW) / b.eta
				B_HP = (HP_FH + HP_HW)
				BD -= B_HP / b.eta
			elseif BD <= (HP_FH + HP_HW) / b.eta
				B_HP = BD * b.eta
				GR_HP = (HP_FH + HP_HW) - B_HP		# slack variable heat pump
			end
		elseif BD <= d_e / b.eta
			B_DE = BD * b.eta
			GR_DE = d_e - B_DE						# slack variable demand
			GR_HP = (HP_FH + HP_HW)					# slack variable heat pump
		end
	end
	################### DETERMINE NEXT STATE ############################
	# Floor heating
	cop_fh = COPcalc(fh, env)
	Hot = IsHot(env)
	T_fh_new = T_fh + ( (60 * 60) / (p_concr * fh.volume * c_concr)) * ( (cop_fh * HP_FH) - d_fh )
	# Calculate loss +/-
	T_fh_new -= ( (60 * 60) / (p_concr * fh.volume * c_concr)) * ( (1 - Hot) * fh.loss - Hot * fh.loss )
	# Comfort violations Floor heating
	if T_fh_new > fh.soc_max
		T_fh_plus = T_fh_new - fh.soc_max
	elseif T_fh_new < fh.soc_min
		T_fh_minus = fh.soc_min - T_fh_new
	end

	# Hot water
	cop_hw = COPcalc(hw, env)
	V_hw_new = V_hw + ((60 * 60) / ( (p_water * hw.t_supply * c_water)  / 1000 )) * ( (cop_hw * HP_HW) - d_hw )
	# Calculate loss -
	V_hw_new = V_hw_new - (60 * 60) / ( (p_water * hw.t_supply * c_water)  / 1000 ) * ( hw.loss )
	# Comfort violations Hot water
	if V_hw_new > hw.soc_max
		V_hw_plus = V_hw_new - hw.soc_max
	elseif V_hw_new < hw.soc_min
		V_hw_minus = hw.soc_min - V_hw_new
	end

	################### Next states ############################
	env.state.T_fh = T_fh_new
	env.state.V_hw = V_hw_new
	# Battery
	env.state.Soc_b = (1 - b.loss) * (Soc_b + PV_B - ( (B_DE + B_HP) / b.eta ))
	# Set uncertain parts of next state
	next_state!(env)
	env.step += 1
	env.idx += 1

	################### DETERMINE REWARD ############################

	comfort = - m.comfort_weight * (V_hw_plus + V_hw_minus + T_fh_plus + T_fh_minus)    # comfort violations
	abort = - 0 * finished(env, env.state) *!(env.step==env.maxsteps)   # abort penalty
	env.reward =  (m.p_sell * PV_GR) - (m.p_buy * (GR_DE + GR_HP)) +    # grid revenue/costs
						comfort + abort

	results = hcat(T_fh, V_hw, Soc_b, T_fh_plus, T_fh_minus, V_hw_plus, V_hw_minus,
					env.reward, comfort, abort, cop_fh, cop_hw, PV_DE, B_DE, GR_DE, PV_B, PV_GR,
					PV_HP, GR_HP, B_HP, HP_FH, HP_HW, env.idx, B_switch, B, HP_switch, FH, HW)

	if track == 0
		return env.reward, Vector{Float32}(env.state)
	else
		return env.reward, Vector{Float32}(env.state), results
	end
end

function finished(env::Shems, s′)
	if env.step >= env.maxsteps # only 24 time steps
		return true
	elseif env.state.V_hw > hw.volume # tank volume 200l max
		return true
	elseif env.state.V_hw < 0 # tank volume 0l max
		return true
	elseif env.state.T_fh > fh.t_supply # heating can't exceed supply
		return true
	elseif env.state.T_fh < (0.8*fh.soc_min) # heating can't fall below 15.2°C
		return true
	else
		return false
	end
end

# ------------------------------------------------------------------------

@recipe function f(env::Shems)
  legend := false
  link := :x
  xlims := (0, 2)
  #grid := false
  xticks := nothing
  layout := (3, 1)

  # battery state
  @series begin
	subplot := 1
	seriestype := :bar
	ylims := (b.soc_min, b.soc_max)
	fillcolor := :purple
    return [1], [env.state.Soc_b]
  end

  # battery range
  @series begin
	subplot := 1
	seriestype := :path
	ylims := (b.soc_min, b.soc_max)
	linecolor := :purple
	annotations := [(0.3, (b.soc_max - 0.5), "B: $(round(env.a[2], digits=5))", :top),
					(1.7, (b.soc_max - 0.5), "Hour: $(mod(env.idx-1, 24))", :top),
					(1.7, (b.soc_max - 3), "Reward: $(round(env.reward, digits=2))", :top),
					(1.7, (b.soc_max - 5.5), "Over?: $(finished(env, env.state))", :top)]
	return [0 0; 2 2], [b.soc_min b.soc_max; b.soc_min b.soc_max]
  end

  # floor heating state
  @series begin
	subplot := 2
	seriestype := :bar
	ylims := (0.8*fh.soc_min, fh.t_supply)
	fillcolor := :firebrick
    return [1], [env.state.T_fh]
  end

  # floor heating comfort range
  @series begin
	  subplot := 2
	  seriestype := :path
	  ylims := (0.8*fh.soc_min, fh.t_supply)
	  linecolor := :firebrick
	  if env.a[2] > 0
		  annotations := [(0.3, (fh.t_supply - 1),
						  "FH: $(round(env.a[4], digits=5))", :top)]
	  end
	  return [0 0; 2 2], [fh.soc_min fh.soc_max; fh.soc_min fh.soc_max]
  end

  # hot water state
  @series begin
	subplot := 3
	seriestype := :bar
	ylims := (0, hw.volume)
	fillcolor := :steelblue
    return [1], [env.state.V_hw]
 end

 # hot water comfort range
 @series begin
	 subplot := 3
	 seriestype := :path
	 ylims := (0, hw.volume)
	 linecolor := :steelblue
	 if env.a[2] < 0
		 annotations := [(0.3, (hw.volume - 20),
						 "HW: $(round(env.a[4], digits=5))", :top)]
	 end
	 return [0 0; 2 2], [hw.soc_min hw.soc_max; hw.soc_min hw.soc_max]
 end

end

end
