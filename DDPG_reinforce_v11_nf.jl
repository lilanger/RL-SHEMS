# Parameters and architecture based on:
# https://github.com/msinto93/DDPG/blob/master/train.py
# https://github.com/JuliaReinforcementLearning/ReinforcementLearningZoo.jl/blob/master/src/experiments/rl_envs.jl
using Flux, Printf, Zygote#, CUDA
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

struct OUNoise
  μ
  σ
  θ
  dt
  X
end

# Critic model
struct crit
  state_crit
  act_crit
  sa_crit
end

function (c::crit)(state, action)
  s = c.state_crit(state)
  a = c.act_crit(action)
  c.sa_crit(relu.(s .+ a))
end

Base.deepcopy(c::crit) = crit(deepcopy(c.state_crit),
							  deepcopy(c.act_crit),
							  deepcopy(c.sa_crit))

NUM_STEPS = 24 #36
NUM_EP = 25 #50_000
L1 = 400
L2 = 300
case = "num_steps=24"
plot_result = false
save_result = false
render_test = true

rng = StableRNG(123)
Random.seed!(123)
start_time = now()
#Load game environment
env = Shems(NUM_STEPS)
reset!(env)

# ----------------------------- Parameters -------------------------------------
STATE_SIZE = length(env.state)
ACTION_SIZE = length(env.a)
ACTION_BOUND_HI = Float32[4.6f0, 3.0f0] #Float32(actions(env, env.state).hi[1])
ACTION_BOUND_LO = Float32[-4.6f0, -3.0f0] #Float32(actions(env, env.state).lo[1])
UPDATE_EVERY = 1

BATCH_SIZE = 64
MEM_SIZE = 1_000_000
MIN_EXP_SIZE = 50_000

γ = 0.99f0     # discount rate for future rewards

τ = 1f-3       # Parameter for soft target network updates
η_act = 1f-4   # Learning rate actor
η_crit = 1f-3  # Learning rate critic
L2_DECAY = 0.01f0

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

# --------------------------------- Memory ------------------------------------
memory = CircularBuffer{Any}(MEM_SIZE)

function getData(batch_size = BATCH_SIZE)
  # Getting data in shape
  minibatch = sample(memory, batch_size)
  x = hcat(minibatch...)

  s       =   hcat(x[1, :]...) |> gpu
  a       =   hcat(x[2, :]...) |> gpu
  r       =   hcat(x[3, :]...) |> gpu
  s_prime =   hcat(x[4, :]...) |> gpu
  s_mask  = .!hcat(x[5, :]...) |> gpu

  return s, a, r, s_prime, s_mask
end

# ------------------------------- Action Noise --------------------------------
# Fill struct with values
ou = OUNoise(μ, θ, σ, dt, zeros(Float32, ACTION_SIZE))

function sample_noise(ou::OUNoise)
  dx     = ou.θ * (ou.μ .- ou.X) * ou.dt
  dx   .+= ou.σ * sqrt(ou.dt) * randn(rng, Float32, length(ou.X))
  return ou.X .+= dx
end

# ----------------------------- Model Architecture -----------------------------
init = Flux.glorot_uniform(rng)
init_final(dims...) = 6f-3rand(rng, Float32, dims...) .- 3f-3

actor = Chain(
			Dense(STATE_SIZE, L1, relu, initW=init, initb=init),
	      	Dense(L1, L2, relu; initW=init, initb=init),
            Dense(L2, ACTION_SIZE, tanh; initW=init_final, initb=init_final)
			) |> gpu
actor_target = deepcopy(actor)

Flux.@functor crit

critic = crit(Chain(Dense(STATE_SIZE, L1, relu, initW=init, initb=init),Dense(L1, L2, initW=init, initb=init)) |> gpu,
              Dense(ACTION_SIZE, L2, initW=init, initb=init) |> gpu,
	      	  Dense(L2, 1, initW=init_final, initb=init_final) |> gpu
			  )
critic_target = deepcopy(critic)
# ---------------------- Param Update Functions --------------------------------
function update_actor!(s)
  grads = gradient(()->loss_act(s), params(actor))
  update!(opt_act, params(actor), grads)
end

function update_critic!(y, s, a)
  grads = gradient(()->loss_crit(y, s, a), params(critic))
  update!(opt_crit, params(critic), grads)
end

function update_actor_target!(;τ = 1f0)
  for (p_t, p_m) in zip(params(actor_target), params(actor))
	p_t .= (1f0 - τ) * p_t .+ τ * p_m
  end
end

function update_critic_target!(;τ = 1f0)
  for (p_t, p_m) in zip(params(critic_target), params(critic))
	p_t .= (1f0 - τ) * p_t .+ τ * p_m
  end
end

# ---------------------------------- Training ----------------------------------
# Losses
function L2_loss(model)
  l2_loss = sum(map(p->sum(p.^2), params(model)))
  return L2_DECAY * l2_loss
end

loss_crit(y, s, a) = Flux.mse(critic(s, a), y) + L2_loss(critic) # L2 loss have huge pos. impact

function loss_act(s)
  s_norm = normalize(cpu(s))
  actions = actor(s_norm |> gpu)
  crit_out = critic(s_norm |> gpu, actions)
  return -sum(crit_out)  # sum better than mean
end

# Optimizers
opt_crit = ADAM(η_crit)
opt_act  = ADAM(η_act)

function replay()
  s, a, r, s′, s_mask = getData()
  update_actor!(s)
  # update Critic
  a′ = actor_target(s′)
  v′ = critic_target(s′, a′)
  y = r .+ (γ * v′ .* s_mask)	# set v′ to 0 where s_ is terminal state
  update_critic!(y, s, a)
  #Update Target models
  update_actor_target!(τ = τ)
  update_critic_target!(τ = τ)
  return nothing
end
# ---------------------------- Helper Functions --------------------------------
# Stores tuple of state, action, reward, next_state, and done
remember(state, action, reward, next_state, done) =
	push!(memory, [state, action, reward, next_state, done])

# Choose action according to policy
function action(;train=true)
	local s_norm = normalize(Flux.unsqueeze(env.state, 2) |> cpu)
	act_pred = actor(s_norm |> gpu) |> cpu
	act_pred += train .* noise_scale .* sample_noise(ou) # add noise only in training
	return act_pred #returns action, assumes ACTION_SIZE=1
end

function episode!(; train=true, render=false)
  reset!(env)
  reward_eps=0f0
  for ep=1:NUM_STEPS
	s = copy(env.state)
    a = action(train=train)
	scaled_action = Float32.(ACTION_BOUND_LO .+ (a .+ ones(ACTION_SIZE)) .* 0.5 .* (ACTION_BOUND_HI .- ACTION_BOUND_LO)) #scale to action range
    r, s_prime = step!(env, s, scaled_action)
	reward_eps += r
	finished(env, s_prime) && break
	if render == true
		sleep(1f-1)
		gui(plot(env))
	end
	if train == true
      remember(s, a, r, s_prime, finished(env, s_prime))
	  if ep % UPDATE_EVERY == 0
		  replay()
	  end
    # elseif train == false
	#   println(s)
	#   println(scaled_action)
	#   println(r)
	#   println(s_prime)
    end
  end
  return (reward_eps / NUM_STEPS)
end
# -------------------------------- Testing -------------------------------------

# Returns average score (per step) over 100 episodes
function test(env::Shems; render=false)
  reward = 0f0
  for e=1:100
  	reward += episode!(train=false, render=render)
  end
  return (reward / 100.)
end


function plot_scores(;total_reward=total_reward, score_mean=score_mean)
	scatter(1:NUM_EP, [total_reward], label="train", left_margin = 12mm,
			markershape=:circle, markersize=3, markerstrokewidth=0.1,
			legend=:bottomright)
	scatter!(1:10:NUM_EP, score_mean, label="test", left_margin = 12mm,
			markershape=:circle, legend=:bottomright, markersize=3, markerstrokewidth=0.1)
	savefig("out/fig/DDPG_Shems_v11_$(NUM_STEPS)_$(NUM_EP)_$(L1)_$(L2)_$(case)")
end

# Populate memory with random actions
function populate_memory(env::Shems)
	while length(memory) < MIN_EXP_SIZE
		reset!(env)
		for e=1:NUM_STEPS
		  s = copy(env.state)
		  a = Float32.(rand(rng, ACTION_SIZE) .* 2 .- 1) # random values between -1 and 1
		  scaled_action = Float32.(ACTION_BOUND_LO .+ (a .+ ones(ACTION_SIZE)) .* 0.5 .* (ACTION_BOUND_HI .- ACTION_BOUND_LO)) #scale to action range
		  r, s_prime = step!(env, s, scaled_action)
		  remember(s, a, r, s_prime, finished(env, s_prime))
		  finished(env, s_prime) && break
		end
	end
	return nothing
end

# --------------------------------- Data preprocessing ------------------------------------
function min_max_buffer(MIN_EXP_SIZE=MIN_EXP_SIZE)
	s, a, r, s_prime, s_mask = getData(MIN_EXP_SIZE)
	return minimum(cpu(s), dims=2), maximum(cpu(s), dims=2)
end

function normalize(s; s_min=s_min, s_max=s_max)
	s = (s .- s_min) ./ (s_max .- s_min .+ 1f-8)
	return s
end

populate_memory(env::Shems)
s_min, s_max = min_max_buffer(MIN_EXP_SIZE)

# ------------------------------ Training --------------------------------------
total_reward = zeros(Float32, NUM_EP)
score_mean = zeros(Float32, ceil(Int32, NUM_EP/10))
if render_test == false
	t_start = now()
	print("Max steps: $(NUM_STEPS), Max episodes: $(NUM_EP), Layer 1: $(L1) nodes, Layer 2: $(L2) nodes, ")
	println("Case: $(case), Time to start: $(round(t_start - start_time, Dates.Minute))")

	function run_episodes(env::Shems, total_reward, score_mean, noise_scale)
		reset!(env)
		for i=1:NUM_EP
		  	total_reward[i] = episode!(train=true, render=false)
		  	print("Episode: $i | Mean Step Score: $(@sprintf "%9.3f" total_reward[i]) | ")
		  	if i % 10 == 1
				idx = ceil(Int32, i/10)
		  		score_mean[idx] = test(env)
		  		print("Mean score over 100 test episodes: $(@sprintf "%9.3f" score_mean[idx]) | ")
		  	end
		  	t_elap = round(now()-t_start, Dates.Minute)
		  	println("Time elapsed: $(t_elap)")
		  	noise_scale = ϵ * noise_scale
		end
		return total_reward, score_mean
	end

	run_episodes(env, total_reward, score_mean, noise_scale)

	# ------------------------- Save results ---------------------------------------
    if save_result == true
		actor = cpu(actor)
		actor_target = cpu(actor_target)
		critic = cpu(critic)
		critic_target = cpu(critic_target)

		BSON.@save "out/bson/DDPG_Shems_v11_$(NUM_STEPS)_$(NUM_EP)_$(L1)_$(L2)_$(case)_actor.bson" actor
		BSON.@save "out/bson/DDPG_Shems_v11_$(NUM_STEPS)_$(NUM_EP)_$(L1)_$(L2)_$(case)_actor_target.bson" actor_target
		BSON.@save "out/bson/DDPG_Shems_v11_$(NUM_STEPS)_$(NUM_EP)_$(L1)_$(L2)_$(case)_critic.bson" critic
		BSON.@save "out/bson/DDPG_Shems_v11_$(NUM_STEPS)_$(NUM_EP)_$(L1)_$(L2)_$(case)_critic_target.bson" critic_target
		BSON.@save "out/bson/DDPG_Shems_v11_$(NUM_STEPS)_$(NUM_EP)_$(L1)_$(L2)_$(case)_scores.bson" total_reward score_mean
	end

	if plot_result == true
		plot_scores()
	end

elseif render_test == true
	# ------------------------ Load and render resulting behavior --------------------
	function render_results(env::Shems; NUM_STEPS=NUM_STEPS, NUM_EP=NUM_EP, L1=L1, L2=L2, case=case, plot_result=plot_result)
		BSON.@load "out/bson/DDPG_Shems_v11_$(NUM_STEPS)_$(NUM_EP)_$(L1)_$(L2)_$(case)_actor.bson" actor
		BSON.@load "out/bson/DDPG_Shems_v11_$(NUM_STEPS)_$(NUM_EP)_$(L1)_$(L2)_$(case)_actor_target.bson" actor_target
		BSON.@load "out/bson/DDPG_Shems_v11_$(NUM_STEPS)_$(NUM_EP)_$(L1)_$(L2)_$(case)_critic.bson" critic
		BSON.@load "out/bson/DDPG_Shems_v11_$(NUM_STEPS)_$(NUM_EP)_$(L1)_$(L2)_$(case)_critic_target.bson" critic_target
		if plot_result == true
			BSON.@load "out/bson/DDPG_Shems_v11_$(NUM_STEPS)_$(NUM_EP)_$(L1)_$(L2)_$(case)_scores.bson" total_reward score_mean
			plot_scores()
		end
		test(env, render=render_test)
		return nothing
	end
	gr()
	render_results(env; NUM_STEPS=NUM_STEPS, NUM_EP=NUM_EP, case=case, plot_result=plot_result)
end


#DDPG_shems(NUM_STEPS = 1, NUM_EP = 1, save_result = false)
#DDPG_shems()

# Render test results
 #DDPG_shems(NUM_STEPS = 3, NUM_EP = 500, L1=400, L2=300, case = "init", render_test = true, plot_result = true)
# DDPG_shems(NUM_STEPS = 1000, NUM_EP = 50000, L1=64, L2=64, case = "cns_init-3_max-steps", render_test = true, plot_result = true)
