# Parameters and architecture based on:
# https://github.com/msinto93/DDPG/blob/master/train.py
# https://github.com/JuliaReinforcementLearning/ReinforcementLearningZoo.jl/blob/master/src/experiments/rl_envs.jl
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
using CSV
gr()

NUM_STEPS = 24 #36
NUM_EP = 3_000 #50_000
L1 = 300 #400
L2 = 600 #300
case = "Yu_input"
plot_result = true
save_result = true
render_test = false

rng = StableRNG(123)
Random.seed!(123)
start_time = now()

#Load game environment
env = Shems(NUM_STEPS)
reset!(env)

# ----------------------------- Environment Parameters -------------------------
STATE_SIZE = length(env.state)
ACTION_SIZE = length(env.a)
ACTION_BOUND_HI = Float32[4.6f0, 3.0f0] #Float32(actions(env, env.state).hi[1])
ACTION_BOUND_LO = Float32[-4.6f0, -3.0f0] #Float32(actions(env, env.state).lo[1])

# --------------------------------- Memory ------------------------------------
BATCH_SIZE = 120 #64
MEM_SIZE = 24_000 #1_000_000
MIN_EXP_SIZE = 24_000 #50_000

memory = CircularBuffer{Any}(MEM_SIZE)

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

L2_DECAY = 0.01f0
UPDATE_EVERY = 1

# Critic model
struct crit
  state_crit
  act_crit
  sa_crit
end

init = Flux.glorot_uniform(rng)
init_final(dims...) = 6f-3rand(rng, Float32, dims...) .- 3f-3

# Optimizers
opt_crit = ADAM(η_crit)
opt_act  = ADAM(η_act)
