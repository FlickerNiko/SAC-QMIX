
# SAC-QMIX

## Requirements

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

## Performance
