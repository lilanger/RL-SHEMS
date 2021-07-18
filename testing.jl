# Parameters and architecture based on:
# https://github.com/msinto93/DDPG/blob/master/train.py
# https://github.com/JuliaReinforcementLearning/ReinforcementLearningZoo.jl/blob/master/src/experiments/rl_envs/JuliaRL_DDPG_Pendulum.jl
# https://github.com/FluxML/model-zoo/blob/master/contrib/games/differentiable-programming/pendulum/DDPG.jl
# -------------------------------- Testing -------------------------------------
# runs steps through data set without reset, rendering decisions
function inference(env::Shems; render=false, track=0, idx=NUM_EP, rng_inf=rng_run)
  local reward = 0f0
  local noise=0f0
  # tracking flows
  if track != 0
    reward, results = episode!(env, NUM_STEPS=EP_LENGTH[season, run], train=false,
                                render=render, track=track, rng_ep=-1)
    write_to_results_file(results, idx=idx, rng=rng_inf)
    return nothing
  # rendering flows
  elseif render == true && track == 0
    reward, noise = episode!(env, NUM_STEPS=EP_LENGTH[season, run], train=false,
                                render=true, track=track, rng_ep=-1)
    return nothing
  # return mean reward over 100 eps
  else
    local runs=100
    local reward = zeros(runs)
    for e = 1:runs
        reward[e], noise = episode!(env, NUM_STEPS=EP_LENGTH[season, run], train=false,
                                        render=render, track=track, rng_ep=e)
    end
    return (reward / runs)
  end
end
