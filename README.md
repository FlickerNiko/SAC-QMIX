Skip to content
Search or jump to…
Pull requests
Issues
Marketplace
Explore
 
@FlickerNiko 
oxwhirl
/
smac
Public
19
598
159
Code
Issues
8
Pull requests
1
Actions
Projects
Wiki
Security
Insights
smac/README.md
@samvelyan
samvelyan Update README.md
Latest commit 371df71 on 29 Jul
 History
 4 contributors
@samvelyan@tabzraz@Zymrael@richardliaw
197 lines (123 sloc)  10.1 KB
   
```diff
- Please pay attention to the version of SC2 you are using for your experiments. 
- Performance is *not* always comparable between versions. 
- The results in SMAC (https://arxiv.org/abs/1902.04043) use SC2.4.6.2.69232 not SC2.4.10.
```

# SAC-QMIX

## Dependency

SMAC, pytorch(GPU support recommanded while training), tensorboard, StarCraft II.
For the installation of SMAC and StarCraft II, refer to the repository of [SMAC](https://github.com/oxwhirl/smac).

## Train

Train a model with the following command:

```shell
python main.py
```

Configs and parameters of the training are defined in `config.json`. Models will be saved at `./models`

## Test

Test a trained model with the following command:

```shell
python test_model.py
```
Configs and parameters of the testing are defined in `test_config.json`. Match the `run_name` items in `config.json` and `test_config.json`.

## Theory & Algorithm

### Architecture

### Computation Flow

##Performance



© 2021 GitHub, Inc.
Terms
Privacy
Security
Status
Docs
Contact GitHub
Pricing
API
Training
Blog
About
