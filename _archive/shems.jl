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

import Reinforce: reset!, actions, finished, step!, state

export
  Shems,
  reset!,
  step!,
  actions,
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
    price::Float64
	comfort_weight::Float64
end

# PV(eta)
pv = PV(0.95f0);
# HeatPump(rate_max)
hp = HeatPump(3.0f0);
# ThermalStorage(volume, loss, t_supply, soc_min, soc_max)
fh = ThermalStorage(10.0f0, 0.045f0, 30.0f0, 20.0f0, 22.0f0);
hw = ThermalStorage(200.0f0, 0.035f0, 45.0f0, 20.0f0, 180.0f0);
# Battery(eta, soc_min, soc_max, rate_max, loss)
b = Battery(0.98f0, 0.0f0, 10.0f0, 4.6f0, 0.00003f0);
# Market(price)
m = Market(0.3f0, 1f0)

const p_concr = 2400.0;   # kg/m^3
const c_concr = 1.0;      # kJ/(kg*°C)
const p_water = 997.0;    # kg/m^3
const c_water = 4.184;    # kJ/(kg*°C)

mutable struct ShemsState{T<:AbstractFloat} <: AbstractVector{T}
  Soc_b::T
  T_fh::T
  V_hw::T
  d_e::T
  d_fh::T
  d_hw::T
  g_e::T
  t_out::T
  h::Int64
  p::T
end

ShemsState() = ShemsState(0f0, 22f0, 180f0, 0f0, 0f0, 0f0, 0f0, 0f0, 1, 0.3f0)

Base.size(::ShemsState) = (10,)

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
								ifelse(i == 9, s.h,
									s.p)))))))))
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
												ifelse(i == 9, :h,
				  								:p))))))))), x)
end

mutable struct ShemsAction{T<:AbstractFloat} <: AbstractVector{T}
  B::T
  HP::T
end

ShemsAction() = ShemsAction(0f0, 0f0)

Base.size(::ShemsAction) = (2,)

function Base.getindex(a::ShemsAction, i::Int)
  (i > length(a)) && throw(BoundsError(a, i))
  ifelse(i == 1, a.B, a.HP)
end

function Base.setindex!(a::ShemsAction, x, i::Int)
  (i > length(a)) && throw(BoundsError(a, i))
  setproperty!(a, ifelse(i == 1, :B, :HP), x)
end

mutable struct Shems{V<:AbstractVector, W<:AbstractVector} <: AbstractEnvironment
  state::V
  reward::Float64
  a::W
  steps::Int
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
			ifelse(i == 4, env.steps,
				ifelse(i == 5, env.maxsteps,
					ifelse(i == 6, env.idx,
						env.path))))))
end

function Base.setindex!(env::Shems, x, i::Int)
  (i > length(env)) && throw(BoundsError(env, i))
  setproperty!(env, ifelse(i == 1, :state,
  				  	ifelse(i == 2, :reward,
				  		ifelse(i == 3, :a,
				  			ifelse(i == 4, :steps,
				  				ifelse(i == 5, :maxsteps,
									ifelse(i == 6, :idx,
				  						:path)))))), x)
end

function COPcalc(ts::ThermalStorage, env::Shems)
    # Calculate coefficients of performance for time period
    return max(5.8 -(1/14 * abs(ts.t_supply - env.state.t_out)), 0);
end

function IsHot(env::Shems)
    # Determine if outside temperture is hotter than inside temperture
    return env.state.t_out > env.state.T_fh
end

Shems(maxsteps=36, path="data/summer_training.csv") =
	Shems(ShemsState(), 0.0, ShemsAction(), 0, maxsteps, 1, path)

function reset!(env::Shems)
  idx = reset_state!(env)
  env.reward = 0.0
  env.a = ShemsAction()
  env.steps = 0
  env.idx = idx
  env.path = env.path
  return env
end

function reset_state!(env::Shems)
	env.state.Soc_b = rand(Uniform(b.soc_min, b.soc_max))
	env.state.T_fh = rand(Uniform(fh.soc_min, fh.soc_max))
	env.state.V_hw = rand(Uniform(hw.soc_min, hw.soc_max))

	df = CSV.read(env.path, DataFrame)
	idx = rand(1:(nrow(df) - env.maxsteps))
	env.state.d_e = df[idx, :electkwh]
	env.state.d_fh = df[idx,:heatingkwh]
    env.state.d_hw = df[idx,:hotwaterkwh]
	env.state.g_e = df[idx,:PV_generation]
	env.state.t_out = df[idx,:Temperature]
	env.state.h = df[idx,:hour]
	env.state.p = m.price
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
	env.state.p = m.price
	return nothing
end

#=
Base.size(ins::IntervalSet{T}) where T <: AbstractVector = (2,)

function Base.getindex(ins::IntervalSet{T} where T <: AbstractVector , i::Int)
  (i > length(ins)) && throw(BoundsError(ins, i))
  ifelse(i == 1, ins.lo, ins.hi)
end

function Base.setindex!(ins::IntervalSet{T} where T <: AbstractVector , x, i::Int)
  (i > length(ins)) && throw(BoundsError(ins, i))
  setproperty!(ins, ifelse(i == 1, :lo,:hi), x)
end
=#
#Base.iterate(ins::IntervalSet{T}) where T <: AbstractVector = Float64[ins.hi[i], ins.lo[i] for i=1:length(ins)]
#Base.iterate(ins::IntervalSet{T}) where T <: AbstractVector = ins.lo, (ins.lo, ins.hi)

actions(env::Shems, s) = IntervalSet([-b.rate_max, -hp.rate_max], [b.rate_max, hp.rate_max])

function step!(env::Shems, s, a; track=false)
	Soc_b, T_fh, V_hw, d_e, d_fh, d_hw, g_e, t_out, h, p = env.state
	B, HP = a
	env.a = ShemsAction(B, HP)
  	pv_, BD, BC, T_fh_plus, T_fh_minus, V_hw_plus, V_hw_minus = zeros(7)
	PV_DE, PV_B, PV_HP, PV_GR, B_HP, GR_HP, HP_FH, HP_HW, B_GR = zeros(9)

	############# DETERMINE FLOWS ###################################
	if B < 0 # battery discharging, restrictions dischraging rate and soc
		BD = abs( clamp(B, max(-b.rate_max,  -((1 - b.loss - 1f-7) *Soc_b)), 0) )
	end

	if HP > 0 # floor heating
		HP_FH = clamp(HP, 0, hp.rate_max)
	elseif HP < 0 # hot water
		HP_HW = abs( clamp(HP, -hp.rate_max, 0) )
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
				B_GR = BD * b.eta
			elseif BD <= (HP_FH + HP_HW - PV_HP) / b.eta
				B_HP = BD * b.eta
				GR_HP = (HP_FH + HP_HW - PV_HP) - B_HP
			end
		end

		# battery charging (only from PV, has to be adapted for dynamic prices!)
		if B > 0
			PV_B = clamp(B, 0, min(pv_, b.rate_max, b.soc_max - Soc_b) )
			pv_ -= PV_B
		end
		PV_GR = pv_ # grid feed-in

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
				B_GR = BD * b.eta
			elseif BD <= (HP_FH + HP_HW) / b.eta
				B_HP = BD * b.eta
				GR_HP = (HP_FH + HP_HW) - B_HP
			end
		elseif BD <= d_e / b.eta
			B_DE = BD * b.eta
			GR_DE = d_e - B_DE
		end
	end
	################### DETERMINE NEXT STATE ############################
	# Floor heating
	if HP_FH > 0
		cop_fh = COPcalc(fh, env)
		T_fh += ( (60 * 60) / (p_concr * fh.volume * c_concr)) * ( (cop_fh * HP_FH) - d_fh )
	# Hot water
	elseif HP_HW > 0
		cop_hw = COPcalc(hw, env)
		V_hw += ((60 * 60) / ( (p_water * hw.t_supply * c_water)  / 1000 )) * ( (cop_hw * HP_HW) - d_hw )
	end
	# Comfort violations Floor heating
	# Calculate loss +/-
	Hot = IsHot(env)
	T_fh -= ( (60 * 60) / (p_concr * fh.volume * c_concr)) * ( (1 - Hot) * fh.loss - Hot * fh.loss )
	if T_fh > fh.soc_max
		T_fh_plus = T_fh - fh.soc_max
	elseif T_fh < fh.soc_min
		T_fh_minus = fh.soc_min - T_fh
	end
	env.state.T_fh = T_fh

	# Comfort violations Hot water
	# Calculate loss -
	V_hw -= (60 * 60) / ( (p_water * hw.t_supply * c_water)  / 1000 ) * ( hw.loss )
	if V_hw > hw.soc_max
		V_hw_plus = V_hw - hw.soc_max
	elseif V_hw < hw.soc_min
		V_hw_minus = hw.soc_min - V_hw
	end
	env.state.V_hw = V_hw
	# Battery
	env.state.Soc_b = ( (1 - b.loss) * Soc_b ) + PV_B - ( (B_DE + B_HP + B_GR) / b.eta_b )
	# Set uncertain parts of next state
	next_state!(env)
	env.steps += 1
	env.idx += 1
	################### DETERMINE REWARD ############################
	env.reward =  (p / 3.) * (PV_GR + B_GR) - (p * (GR_DE + GR_HP)) -    # grid revenue/costs
					m.comfort_weight * (V_hw_plus + V_hw_minus + T_fh_plus + T_fh_minus) #-    # comfort violations
						#finished(env, env.state) * 100    # abort penalty
	if track == false
		return env.reward, Vector{Float32}(env.state)
	elseif track == true
		results =
			hcat(T_fh, V_hw, Soc_b, T_fh_plus, T_fh_minus, V_hw_plus, V_hw_minus,
					env.reward, cop_fh, cop_hw, PV_DE, B_DE, GR_DE, PV_B, PV_GR,
					PV_HP, GR_HP, B_HP, HP_FH, HP_HW, env.state.h, env.idx)
		return env.reward, Vector{Float32}(env.state), results
	end
end

# function state(env::Shems)
#   Soc_b, T_fh, V_hw, d_e, d_fh, d_hw, pv, t_out, h, p = env.state
#   return Float64[Soc_b, T_fh, V_hw, d_e, d_fh, d_hw, pv, t_out, h, p]
# end

#finished(env::Shems, s′) = env.steps >= env.maxsteps

function finished(env::Shems, s′)
	if env.steps >= env.maxsteps # only 36 time steps
		return true
	elseif env.state.V_hw > hw.volume # tank volume 200l max
		return true
	elseif env.state.V_hw < 0 # tank volume 0l max
		return true
	elseif env.state.T_fh > fh.t_supply # heating can't exceed supply
		return true
	elseif env.state.T_fh < (0.8*fh.soc_min) # heating can't fall below 16°C
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
	annotations := [(0.3, (b.soc_max - 0.5), "B: $(round(env.a[1], digits=5))", :top),
					(1.7, (b.soc_max - 0.5), "Hour: $(env.state.h)", :top),
					(1.7, (b.soc_max - 3), "Reward: $(round(env.reward, digits=2))", :top)]
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
						  "FH: $(round(env.a[2], digits=5))", :top)]
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
						 "HW: $(round(abs(env.a[2]), digits=5))", :top)]
	 end
	 return [0 0; 2 2], [hw.soc_min hw.soc_max; hw.soc_min hw.soc_max]
 end

end

end
