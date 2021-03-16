# Parameters and architecture based on:
# https://github.com/msinto93/DDPG/blob/master/train.py
# https://github.com/JuliaReinforcementLearning/ReinforcementLearningZoo.jl/blob/master/src/experiments/rl_envs/JuliaRL_DDPG_Pendulum.jl
# https://github.com/FluxML/model-zoo/blob/master/contrib/games/differentiable-programming/pendulum/DDPG.jl
# -------------------------------- Testing -------------------------------------

# Returns average score (per step) over 100 episodes
function test(env::Shems; render=false, track=false)
  reward = 0f0
  for e=1:100
  	reward += episode!(env_eval, train=false, render=render, track=track)
  end
  return (reward / 100.)
end

# runs steps through data set without reset, rendering decisions
function inference(;render=false, track=false)
  reward = 0f0
  learn_steps = NUM_STEPS
  global NUM_STEPS = 383 # whole evaluation set 1 month
  env_eval = Shems(NUM_STEPS, "data/$(season)_evaluation.csv")
  # tracking flows
  if track == true
    reward, results = episode!(env_eval, train=false, render=render, track=true)
    write_to_results_file(results, learn_steps)
    return nothing
  # rendering flows
  elseif render == true && track == false
    reward = episode!(env_eval, train=false, render=true, track=false)
    return nothing
  # return mean reward over 100 eps
  else
    for e = 1:100
        reward += episode!(env_eval, train=false, render=render, track=track)
    end
    return (reward / 100.)
  end
end
