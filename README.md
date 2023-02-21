# Dynamic Graph-based Communication Mechanism for Value Factorization Multi-Agent Learning

This is the implementation of the paper "Dynamic Graph-based Communication Mechanism for Value Factorization Multi-Agent Learning"

This implementation is written in PyTorch and is based on [PyMARL](https://github.com/oxwhirl/pymarl), [PyMARL2](https://github.com/hijkzzz/pymarl2) and [SMAC](https://github.com/oxwhirl/smac).

## Installation instructions

Install and setup StarCraft II and SMAC:

```shell
bash install_sc2.sh
```

This will download SC2 (4.10) into the 3rdparty folder and copy the maps necessary to run over.

```diff
- Please pay attention to the version of SC2 you are using for your experiments. 
- Performance is *not* always comparable between versions. 
- The results in the paper use SC2.4.10.
```

Install Python packages:

```shell
pip install -r requirements.txt
```

**NOTE**: Before you run a DGCM-based experiment, please first modify the `starcraft.py` file to enable an `adjacency matrix` function, then modify the runner and replaybuffer files to store data. 


## Run an experiment

```shell
python3 src/main.py --config=dgcm_qmix --env-config=sc2 with env_args.map_name=5m_vs_6m
```

All results will be stored in the Results folder.

## Saving and loading learnt models

### Saving models

You can save the learnt models to disk by setting `save_model = True`, which is set to False by default. The frequency of saving models can be adjusted using `save_model_interval` configuration. Models will be saved in the result directory, under the folder called models. The directory corresponding each run will contain models saved throughout the experiment, each within a folder corresponding to the number of timesteps passed since starting the learning process.

### Loading models

Learnt models can be loaded using the `checkpoint_path` parameter, after which the learning will proceed from the corresponding timestep.

## Watching StarCraft II replays

`save_replay` option allows saving replays of models which are loaded using `checkpoint_path`. Once the model is successfully loaded, `test_nepisode` number of episodes are run on the test mode and a .SC2Replay file is saved in the Replay directory of StarCraft II. Please make sure to use the episode runner if you wish to save a replay, i.e., `runner=episode`. The name of the saved replay file starts with the given `env_args.save_replay_prefix` (map_name if empty), followed by the current timestamp.

**Note**: Replays cannot be watched using the Linux version of StarCraft II. Please use either the Mac or Windows version of the StarCraft II client.
