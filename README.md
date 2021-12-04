
# SAC-QMIX

Algorithm that applies SAC to QMIX for Multi-Agent Reinforcement Learning. Watch the [demo](https://youtu.be/T0t-d1e7IkE).

## Requirements

SMAC

pytorch (GPU support recommanded while training)

tensorboard

StarCraft II

For the installation of SMAC and StarCraft II, refer to the repository of [SMAC](https://github.com/oxwhirl/smac).

## Train

Train a model with the following command:

```shell
python main.py
```

Configurations and parameters of the training are specified in `config.json`. Models will be saved at `./models`

## Test

Test a trained model with the following command:

```shell
python test_model.py
```
Configurations and parameters of the testing are specified in `test_config.json`. Match the `run_name` items in `config.json` and `test_config.json`.

## Theory & Algorithm

### Architecture

<div align=center><img src = "https://github.com/FlickerNiko/SAC-QMIX/blob/master/SAC-QMIX.svg"/></div>

### Computation Flow

Note that a_i is equivalent to \mu_i and s_i is equivalent to o_i in the architecture schema above.

Train Objective: policies that maximum

<img src="https://github.com/FlickerNiko/SAC-QMIX/blob/master/formulas/formula1.png"/>

Q-values computed by networks: 

<img src="https://github.com/FlickerNiko/SAC-QMIX/blob/master/formulas/formula2.png"/>

Individual state-value functions: 

<img src="https://github.com/FlickerNiko/SAC-QMIX/blob/master/formulas/formula4.png"/>

Total state-values (alpha is the entropy temperature):

<img src="https://github.com/FlickerNiko/SAC-QMIX/blob/master/formulas/formula3.png"/>

Q-values expressed with Bellman Function: 

<img src="https://github.com/FlickerNiko/SAC-QMIX/blob/master/formulas/formula5.png"/>

Critic networks update: minimum

<img src="https://github.com/FlickerNiko/SAC-QMIX/blob/master/formulas/formula6.png"/>

Actor networks update: maximum

<img src="https://github.com/FlickerNiko/SAC-QMIX/blob/master/formulas/formula7.png"/>

Entropy temperatures update: minimum

<img src ="https://github.com/FlickerNiko/SAC-QMIX/blob/master/formulas/formula8.png"/>





## Result

Note that data of other algorithm are from [SMAC paper](https://github.com/oxwhirl/smac/releases/download/v1/smac_run_data.json). Therefore methods of evaluations are kept the same as [SMAC paper](https://arxiv.org/abs/1902.04043) did (StarCraftII version: SC2.4.6.2.69232). 

### Test Win Rate % of SAC-QMIX and other algorithms

(Mean of 5 independent runs)

<div align=center>
  
|  Scenario  | IQL | VDN | QMIX | SAC-QMIX |
|  :-------: | :-: | :-: | :--: | :------: |
|  2s_vs_1sc | 100 | 100 | 100  | 100 |
|  2s3z      | 75  | 97  | 99   | 100 |
|  3s5z      | 10  | 84  | 97   | 97  |
|  1c3s5z    | 21  | 91  | 97   | 100 |
| 10m_vs_11m | 34  | 97  | 97   | 100 |
| 2c_vs_64zg | 7   | 21  | 58   | 56  |
|bane_vs_bane| 99  | 94  | 85   | 100 |
|  5m_vs_6m  | 49  | 70  | 70   | 90  |
|  3s_vs_5z  | 45  | 91  | 87   | 100 |
|3s5z_vs_3s6z| 0   | 2   | 2    | 85  |
|  6h_vs_8z  | 0   | 0   | 3    | 82  |
| 27m_vs_30m | 0   | 0   | 49   | 100 |
|   MMM2     | 0   | 1   | 69   | 95  |
|  corridor  | 0   | 0   | 1    | 0   |
</div>

### Learning curves of SAC-QMIX and other algorithms

(Mean of 5 independent runs)

<div align=center><img src ="https://github.com/FlickerNiko/SAC-QMIX/blob/master/figures/5m_vs_6m_all.svg"/></div>
<div align=center><img src ="https://github.com/FlickerNiko/SAC-QMIX/blob/master/figures/27m_vs_30m_all.svg"/></div>
<div align=center><img src ="https://github.com/FlickerNiko/SAC-QMIX/blob/master/figures/2c_vs_64zg_all.svg"/></div>
<div align=center><img src ="https://github.com/FlickerNiko/SAC-QMIX/blob/master/figures/MMM2_all.svg"/></div>
<div align=center><img src ="https://github.com/FlickerNiko/SAC-QMIX/blob/master/figures/3s5z_vs_3s6z_all.svg"/></div>
<div align=center><img src ="https://github.com/FlickerNiko/SAC-QMIX/blob/master/figures/6h_vs_8z_all.svg"/></div>
