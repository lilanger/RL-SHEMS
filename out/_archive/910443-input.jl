# Parameters and architecture based on:
# https://github.com/msinto93/DDPG/blob/master/train.py
# https://github.com/JuliaReinforcementLearning/ReinforcementLearningZoo.jl/blob/master/src/experiments/rl_envs/JuliaRL_DDPG_Pendulum.jl
# https://github.com/FluxML/model-zoo/blob/master/contrib/games/differentiable-programming/pendulum/DDPG.jl
using Pkg
Pkg.activate(pwd())
using Flux, Printf, Zygote, CUDA
using Flux.Optimise: update!
using BSON
using Statistics: mean
using DataStructures: CircularBuffer
using Distributions: sample
using Random, StableRNGs
using Reinforce
using Reinforce.ShemsEnv: Shems
using Dates
using Plots, Measures
using CSV, DataFrames
gr()

train = 1
plot_result = 1
render = 0
track = 0
season = "winter"
case = "$(season)_no-L2_best_no-day"

NUM_STEPS = 24 #36
NUM_EP = 3_000 #50_000
L1 = 400 #300
L2 = 300 #600

rng = StableRNG(123)
Random.seed!(123)
start_time = now()

# --------------------------------- Memory ------------------------------------
BATCH_SIZE = 120 #64
MEM_SIZE = 24_000 #1_000_000
MIN_EXP_SIZE = 1_200 #24_000 #50_000

memory = CircularBuffer{Any}(MEM_SIZE)

# --------------------------------- Game environment ---------------------------
env = Shems(NUM_STEPS, "data/$(season)_training.csv")
env_eval = Shems(NUM_STEPS, "data/$(season)_evaluation.csv")
env_test = Shems(NUM_STEPS, "data/$(season)_testing.csv")

# ----------------------------- Environment Parameters -------------------------
STATE_SIZE = length(env.state)
ACTION_SIZE = length(env.a)
ACTION_BOUND_HI = Float32[4.6f0, 3.0f0] #Float32(actions(env, env.state).hi[1])
ACTION_BOUND_LO = Float32[-4.6f0, -3.0f0] #Float32(actions(env, env.state).lo[1])

# ------------------------------- Action Noise --------------------------------
struct OUNoise
  μ
  σ
  θ
  dt
  X
end

# Ornstein-Uhlenbeck Noise params
# based on: https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
μ = 0f0 #mu
σ = 0.3f0 #sigma
θ = 0.15f0 #theta
dt = 1f-2

# Noise scale
τ_ = 25
ϵ  = exp(-1f0 / τ_)
noise_scale = 0.1f0

# Fill struct with values
ou = OUNoise(μ, θ, σ, dt, zeros(Float32, ACTION_SIZE))

# ----------------------------- Model Architecture -----------------------------
γ = 0.99f0     # discount rate for future rewards

τ = 1f-3       # Parameter for soft target network updates
η_act = 1f-4   # Learning rate actor
η_crit = 1f-3  # Learning rate critic

#L2_DECAY = 0.01f0
UPDATE_EVERY = 1

init = Flux.glorot_uniform(rng)
init_final(dims...) = 6f-3rand(rng, Float32, dims...) .- 3f-3

# Optimizers
#opt_crit = Flux.Optimiser(WeightDecay(L2_DECAY), ADAM(η_crit))
#opt_act = Flux.Optimiser(WeightDecay(L2_DECAY), ADAM(η_act))
opt_crit = ADAM(η_crit)
opt_act = ADAM(η_act)


# ----------------------------- Model Architecture -----------------------------
actor = Chain(
			Dense(STATE_SIZE, L1, relu, init=init),
	      	Dense(L1, L2, relu; init=init),
          	Dense(L2, ACTION_SIZE, tanh; init=init_final)) |> gpu

actor_target = deepcopy(actor)

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

critic = crit(
			Chain(
				Dense(STATE_SIZE, L1, relu, init=init),
				Dense(L1, L2, init=init)) |> gpu,
  			Dense(ACTION_SIZE, L2, init=init) |> gpu,
  			Dense(L2, 1, init=init_final) |> gpu)

critic_target = deepcopy(critic)
