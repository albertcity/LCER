# Learning Controllable Elements OrientedRepresentations for Reinforcement Learning

This repository is the implementation of LCER(Learning Controllable Elements OrientedRepresentations for Reinforcement Learning) for the DeepMind control experiments.
Our implementation of SAC is based on [SAC+AE](https://github.com/denisyarats/pytorch_sac_ae) by Denis Yarats and [CURL](https://github.com/MishaLaskin/curl) 

## Installation 

All of the dependencies are in the `conda_env.yml` file. They can be installed manually or with the following command:

```
conda env create -f conda_env.yml
```

## Instructions

To reproduce the result of Figure.1 in LCER, please run the `train.py` script:
```
python train.py \
    --domain_name cartpole \
    --task_name swingup \
    --action_repeat 4 \
    --init_steps 1000 \
    --init_encoder_steps 4000 \
    --save_tb \
    --work_dir ./tmp \
    --agent lecr_beta_0.1 \
    --num_train_steps 2500000 
```

You should choose different `action_repeat` , `init_steps`, `init_encoder_steps` according to the tasks:

|task|action repeat|init steps| init encoder steps|
|-|-|-|-|
|cartpole-swingup | 4| 1000| 5000|
|cheetah-run,reacher-easy,ball\_in\_cup-catch | 4| 2500| 10000|
|walker-walk| 2| 5000| 10000|
|finger-spin | 1| 10000| 50000|

In your console, you should see printouts that look like:

```
| train | E: 221 | S: 28000 | D: 18.1 s | R: 785.2634 | BR: 3.8815 | A_LOSS: -305.7328 | CR_LOSS: 190.9854 | CU_LOSS: 0.0000
```

Log abbreviation mapping:

```
train - training episode
E - total number of episodes 
S - total number of environment steps
D - duration in seconds to train 1 episode
R - mean episode reward
BR - average reward of sampled batch
A_LOSS - average loss of actor
CR_LOSS - average loss of critic
CU_LOSS - average loss of the LCER encoder
```

All data related to the run is stored in the specified `working_dir`. To enable model or video saving, use the `--save_model` or `--save_video` flags. For all available flags, inspect `train.py`. To visualize progress with tensorboard run:

```
tensorboard --logdir log --port 6006
```

and go to `localhost:6006` in your browser. If you're running headlessly, try port forwarding with ssh. 

For GPU accelerated rendering, make sure EGL is installed on your machine and set `export MUJOCO_GL=egl`. For environment troubleshooting issues, see the DeepMind control documentation.
