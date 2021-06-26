# Parameters and architecture based on:
# https://github.com/msinto93/DDPG/blob/master/train.py
# https://github.com/JuliaReinforcementLearning/ReinforcementLearningZoo.jl/blob/master/src/experiments/rl_envs/JuliaRL_DDPG_Pendulum.jl
# https://github.com/FluxML/model-zoo/blob/master/contrib/games/differentiable-programming/pendulum/DDPG.jl
# -------------------------------- Testing -------------------------------------

# Returns average score (per step) over 100 episodes
function test(env::Shems; render=false, track=0)
  local reward = zeros(100)
  local noise=0f0
  for e=1:100
  	reward[e], noise = episode!(env, NUM_STEPS=EP_LENGTH[season, run], train=false, render=render, track=track, rng=e)
  end
  return [mean(reward), std(reward)]
end

# runs steps through data set without reset, rendering decisions
function inference(env::Shems; render=false, track=0, idx=NUM_EP)
  local reward = 0f0
  local noise=0f0
  # tracking flows
  if track != 0
    reward, results = episode!(env, NUM_STEPS=EP_LENGTH[season, run], train=false, render=render, track=track, rng=-1)
    write_to_results_file(results, idx=idx)
    return nothing
  # rendering flows
elseif render == true && track == 0
    reward, noise = episode!(env, NUM_STEPS=EP_LENGTH[season, run], train=false, render=true, track=track, rng=-1)
    return nothing
  # return mean reward over 10 eps
  else
    for e = 1:100
        reward, noise = episode!(env, NUM_STEPS=EP_LENGTH[season, run], train=false, render=render, track=track, rng=e)
    end
    return (reward / 100.)
  end
end
