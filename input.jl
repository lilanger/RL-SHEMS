# Parameters and architecture based on:
# https://github.com/msinto93/DDPG/blob/master/train.py
# https://github.com/JuliaReinforcementLearning/ReinforcementLearningZoo.jl/blob/master/src/experiments/rl_envs/JuliaRL_DDPG_Pendulum.jl
# https://github.com/FluxML/model-zoo/blob/master/contrib/games/differentiable-programming/pendulum/DDPG.jl
using Pkg
Pkg.activate(pwd())
using Flux, Printf, Zygote
using CUDA
CUDA.allowscalar(true)
using Flux.Optimise: update!
using BSON
using Statistics: mean, std
using DataStructures: CircularBuffer
using Distributions: sample, Normal
using Random
using Reinforce
using Reinforce.ShemsEnv: Shems
using Dates
using Plots
using CSV, DataFrames
gr()

#------------ local machine ----------
Job_ID=2
seed_run=1
Task_ID=1
#--------cluster jobs------------
# Job_ID = ENV["JOB_ID"]
# Task_ID = ENV["SGE_TASK_ID"]
# seed_run = parse(Int, Task_ID)
#-------------------------------- INPUTS --------------------------------------------
train = 1
plot_result = 1
plot_all = 1
render = 0
track = 1  # 0 - off, 1 - DRL, -1 - rule-based 1, -2 rule-based 2

season = "all"
case = "$(season)_no-L2_nns-gn.2_abort"
run = "eval"
NUM_EP = 3_001 #3_001 #50_000
# L1 = 400 #300
# L2 = 300 #600
L1 = 300
L2 = 600
idx=NUM_EP
test_every = 100
test_runs = 100
num_seeds = 40
algo="DDPG"

#-------------------------------------
seed_ini = 123
# individual random seed for each run
rng_run = parse(Int, string(seed_ini)*string(seed_run))


start_time = now()
current_episode = 0

#--------------------------------- Memory ------------------------------------
BATCH_SIZE = 64
MEM_SIZE = 24_000
MIN_EXP_SIZE = 24_000

########################################################################################
memory = CircularBuffer{Any}(MEM_SIZE)

#--------------------------------- Game environment ---------------------------
EP_LENGTH = Dict("train" => 24,
					("summer", "eval") => 359, ("summer", "test") => 767,
					("winter", "eval") => 359, ("winter", "test") => 719,
					("both", "eval") => 719,   ("both", "test") => 1487,
					("all", "eval") => 1439,   ("all", "test") => 2999) # length of whole evaluation set (different)

env_dict = Dict("train" => Shems(EP_LENGTH["train"], "data/$(season)_train.csv"),
				"eval" => Shems(EP_LENGTH[season, "eval"], "data/$(season)_eval.csv"),
				"test" => Shems(EP_LENGTH[season, "test"], "data/$(season)_test.csv"))


# ----------------------------- Environment Parameters -------------------------
STATE_SIZE = length(env_dict["train"].state)
ACTION_SIZE = length(env_dict["train"].a)
#ACTION_BOUND_HI = Float32[1f0, 1f0] #Float32(actions(env, env.state).hi[1])
#ACTION_BOUND_LO = Float32[-1f0, -1f0] #Float32(actions(env, env.state).lo[1])
ACTION_BOUND_HI = Float32[4.6f0, 3.0f0] #Float32(actions(env, env.state).hi[1])
ACTION_BOUND_LO = Float32[-4.6f0, -3.0f0] #Float32(actions(env, env.state).lo[1])

#------------------------------- Action Noise --------------------------------
struct OUNoise
  μ
  σ
  θ
  dt
  X
end

struct GNoise
  μ
  σ
end

mutable struct EpsNoise
  ζ
  ξ
  ξ_min
end

# Ornstein-Uhlenbeck / Gaussian Noise params
# based on: https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
μ = 0f0 #mu
θ = 0.15f0 #theta
σ = 0.2f0 #sigma
dt = 1f-2
# Epsilon Noise parameters based on Yu et al. 2019
ζ = 0.0005f0
ξ_0 = 0.5f0
ξ_min = 0.1f0

# Noise actor
noise_act = 2f-1

# Fill struct with values
ou = OUNoise(μ, σ, θ, dt, zeros(Float32, ACTION_SIZE))
gn = GNoise(μ, noise_act)
en = EpsNoise(ζ, ξ_0, ξ_min)

#----------------------------- Model Architecture -----------------------------
γ = 0.995f0     # discount rate for future rewards #Yu

τ = 1f-3       # Parameter for soft target network updates
η_act = 1f-4   # Learning rate actor 10^(-4)
η_crit = 1f-3  # Learning rate critic

#L2_DECAY = 0.01f0

init = Flux.glorot_uniform(MersenneTwister(rng_run))
init_final(dims...) = 6f-3rand(MersenneTwister(rng_run), Float32, dims...) .- 3f-3

# Optimizers
# with L2 regularization
#opt_crit = Flux.Optimiser(WeightDecay(L2_DECAY), ADAM(η_crit))
#opt_act = Flux.Optimiser(WeightDecay(L2_DECAY), ADAM(η_act))
# without L2 regularization
opt_crit = ADAM(η_crit)
opt_act = ADAM(η_act)

#----------------------------- Model Architecture -----------------------------
actor = Chain(
			Dense(STATE_SIZE, L1, relu, init=init),
	      	Dense(L1, L2, relu; init=init),
          	Dense(L2, ACTION_SIZE, tanh; init=init_final)) |> gpu

actor_target = deepcopy(actor) |> gpu

# Critic model
struct crit
  state_crit
  act_crit
  sa_crit
end

Flux.@functor crit

function (c::crit)(state, action)
  s = c.state_crit(state)
  a = c.act_crit(action)
  c.sa_crit(relu.(s .+ a))
end

Base.deepcopy(c::crit) = crit(deepcopy(c.state_crit),
							  deepcopy(c.act_crit),
							  deepcopy(c.sa_crit))

critic = crit(Chain(Dense(STATE_SIZE, L1, relu, init=init),Dense(L1, L2, init=init)) |> gpu,
  				Dense(ACTION_SIZE, L2, init=init) |> gpu,
  				Dense(L2, 1, init=init_final) |> gpu)

critic_target = deepcopy(critic) |> gpu
