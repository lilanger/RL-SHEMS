# RL-SHEMS
 
This repository belongs to a publication recently accepted by Applied Energy, the publication will be linked here once it is online.

The publication is closely linked to another publication/repository of mine: https://https://github.com/lilanger/SHEMS

Langer, Lissy, and Thomas Volling. "An optimal home energy management system for modulating heat pumps and photovoltaic systems." Applied Energy 278 (2020): 115661. https://doi.org/10.1016/j.apenergy.2020.115661

Preprint available here: https://arxiv.org/abs/2009.02349

This repository takes the model predictive control (MPC) implementation of the above Smart Home Energy Management (SHEMS) system and translates it into a reinforcement learning (RL) environment. The different environments tested are implemented using the Julia package [Reinforce.jl](https://github.com/JuliaML/Reinforce.jl) which is very light-weight.

The RL environment is solved using the deep deterministic policy gradient ([DDPG](https://www.deepmind.com/publications/deterministic-policy-gradient-algorithms)) algorithm implemented using the Julia package [Flux.jl](https://github.com/FluxML/Flux.jl). You will find some other algorithms in the repository but most of them will not work in their current state.

## HOW IT WORKS (some hints)

### Loading the right environment
The repository contains a [Manifest](Manifest) and [Project](Project) files, so that the same Julia package versions can be installed. Julia is used in version 1.6.1.

### On a cluster
When running the model on a cluster the job files can be used, default is for a gpu [(jobfile_ddpg_v12)](jobfile_ddpg_v12) but there is also a cpu version [(jobfile_ddpg_v12_cpu)](jobfile_ddpg_v12_cpu). For example, 40 parallel model runs can then be started using the bash command: qsub -t 1-40:1 jobfile_ddpg_v12.job

### Work flow
- In general, all input data is fed from the [input.jl](input) file, there are some templates for the different algorithms available. The input files of previous runs are saved in [out/input](out/input). 
- The general workflow is defined in [DDPG_reinforce_v12_nf](DDPG_reinforce_v12_nf). 
- The folder [algorithms](algorithms) contains the code for the DDPG implementation. 
- The folder [Reinforce.jl...](Reinforce.jl-%20files%20to%20add%20(envs)%20and%20to%20change) contains the RL environments and the file to embed them in the Reinforce package. The environments used in the paper are Case A: [H10](Reinforce.jl-%20files%20to%20add%20(envs)%20and%20to%20change/envs/shems_H10.jl) , CaseB: [H9](Reinforce.jl-%20files%20to%20add%20(envs)%20and%20to%20change/envs/shems_H9.jl) and Case C: [U8](Reinforce.jl-%20files%20to%20add%20(envs)%20and%20to%20change/envs/shems_U8.jl).
- The [Analysis-cases](Analysis-cases) folder contains the result analysis of the cases run and the results of the main runs illustrated in the paper.
- The [data](data) folder contains the input data of the RL environment.

I tried to add some comments in the code, so other people would be able to understand what is going on. I hope I was somewhat successful.
If you have questions, just raise an issue and I will try to help.

