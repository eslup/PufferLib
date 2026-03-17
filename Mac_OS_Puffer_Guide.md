# Beginner's Guide to PufferLib for Apple Silicon Users v1

This guide is for the local checkout at `PufferLib/` in this workspace.
It is based on the 3.0 code state we actually validated here on Apple
Silicon. It is meant to be a working runbook.

## Scope

This guide answers four practical questions:

1. What is PufferLib, in the smallest useful sense?
2. What is the shortest path to a successful local run?
3. What worked on this machine, with this checkout?
4. What should you do next for Squared and Breakout?

## Latest verified outcomes

These are the concrete things that were verified locally.

- `puffer eval puffer_breakout --train.device cpu` launched the native
  raylib window cleanly.
- The bounded Squared smoke path completed on:
  - CPU + `Serial`
  - CPU + `Multiprocessing`
  - MPS + `Serial`
- `puffer eval puffer_squared --load-model-path latest --train.device cpu`
  loaded and rendered cleanly.
- Squared was pushed to effective mastery locally. The current mastered
  checkpoint scored:
  - greedy success `0.99999` over 100,000 episodes
  - sampled success `0.99606` over 100,000 episodes
- Breakout render works locally, and the default Breakout model family
  trains cleanly on both CPU and MPS.
- A long Breakout mastery run on MPS with `vec.num-envs = 8` and
  `env.num-envs = 4096` finished cleanly at `perf = 1.000`.
- The finished solved Breakout checkpoint is:
  `experiments/puffer_breakout_xxx.pt`.

## What PufferLib is

PufferLib is a reinforcement learning toolkit built around:

- fast environment simulation
- a built-in training loop
- vectorized rollouts
- a CLI-first workflow

The practical value is that you can go from "environment renders" to
"agent trains" without building an entire RL stack first.

The layers a beginner touches first are:

- `puffer` CLI
- Ocean environments
- PuffeRL training loop
- vector backends such as `Serial` and `Multiprocessing`

Ignore sweeps and tuning until train/eval works once.

## The minimum vocabulary you need

You do not need much RL theory to use this repo productively. Start with:

- Agent: the model making decisions
- Environment: the task or simulator
- Observation: what the environment returns each step
- Action: what the agent chooses each step
- Reward: the scalar signal the agent tries to maximize
- Episode: one run until termination or truncation
- Policy: the function or network mapping observations to actions
- Checkpoint: saved model weights
- Evaluation: running the policy without updating it
- Agent steps: steps counted across many agents in parallel

That last one matters because PufferLib reports throughput as `SPS`
and counts many parallel agent interactions at once.

## The two `num_envs` knobs

This is the biggest beginner confusion point in this repo.

There are two different layers that both expose a `num_envs` setting:

- `--vec.num-envs`
  - how many environment instances the vector backend manages
- `--env.num-envs`
  - a parameter passed into the environment itself

For the native Ocean environments we actually used here, the env-level
`num_envs` becomes the number of agents inside each vectorized instance.

So for `puffer_squared` and `puffer_breakout`, a good mental model is:

```text
total parallel agents ~= vec.num-envs * env.num-envs
```

Examples:

- `vec.num-envs = 2`, `env.num-envs = 16` -> about 32 agents
- `vec.num-envs = 8`, `env.num-envs = 4096` -> about 32,768 agents

When the guide talks about "2048 agents" or "32768 agents", this is why.

## Shortest reliable path that works on this machine

These commands assume:

```bash
cd /pufferlib/PufferLib
source .venv/bin/activate
```

### 0. Verify the CLI

```bash
which puffer
puffer --help
puffer train puffer_squared --help
```

Success means:

- `which puffer` points into `.venv`
- help renders without import errors

### 1. Render an environment without training

```bash
puffer eval puffer_breakout --train.device cpu
```

Why this is the right first test:

- it proves the CLI works
- it proves the Ocean environment loads
- it proves the native render path works

Success means:

- a raylib window opens
- the environment updates
- there is no exception spam

How to exit:

- `Ctrl+C` works
- `Escape` inside the Breakout render window also exits

Important: `eval` does not terminate by itself.

### 2. Train with the smallest stable Squared settings first

```bash
puffer train puffer_squared \
  --vec.backend Serial \
  --vec.num-envs 2 \
  --vec.num-workers 2 \
  --vec.batch-size 2 \
  --env.num-envs 16 \
  --train.bptt-horizon 32 \
  --train.minibatch-size 1024 \
  --train.total-timesteps 2048 \
  --train.device cpu
```

Why this is the safest first command:

- `puffer_squared` is simple
- `Serial` removes multiprocessing as a failure source
- `cpu` avoids MPS questions until the baseline is known-good
- the sizes are small enough to finish quickly

Success means:

- the dashboard appears
- the run completes
- a checkpoint is written under `experiments/`

### 3. Evaluate the newest Squared checkpoint

```bash
puffer eval puffer_squared --load-model-path latest --train.device cpu
```

Success means:

- the newest top-level Squared checkpoint is found
- it loads without shape/device errors
- the environment renders

Interrupt it after it starts running.

### 4. Add multiprocessing only after CPU + Serial works

```bash
puffer train puffer_squared \
  --vec.backend Multiprocessing \
  --vec.num-envs 2 \
  --vec.num-workers 2 \
  --vec.batch-size 2 \
  --env.num-envs 16 \
  --train.bptt-horizon 32 \
  --train.minibatch-size 1024 \
  --train.total-timesteps 2048 \
  --train.device cpu
```

This was verified locally.

Do not change backend and device at the same time while debugging.

### 5. Then test MPS with the same bounded setup

```bash
puffer train puffer_squared \
  --vec.backend Serial \
  --vec.num-envs 2 \
  --vec.num-workers 2 \
  --vec.batch-size 2 \
  --env.num-envs 16 \
  --train.bptt-horizon 32 \
  --train.minibatch-size 1024 \
  --train.total-timesteps 2048 \
  --train.device mps
```

This also completed successfully on this machine.

## Apple Silicon guidance

The practical local recommendation is:

1. CPU + `Serial` first
2. CPU + `Multiprocessing` second
3. MPS benchmark third

Why this ordering is still the right default here:

- the global default device is CUDA-oriented, so macOS users must override
  `--train.device`
- parts of the trainer still assume CUDA/NCCL for distributed workflows
- the dashboard GPU stats are CUDA-only and will show `0` on MPS
- the custom advantage path uses a CPU fallback when CUDA/HIP fast paths are
  not available

That does not mean MPS is bad. It means CPU is the easiest first success path.

### What happened locally on this machine

Small bounded validation run:

- CPU + `Serial` completed cleanly
- CPU + `Multiprocessing` completed cleanly
- MPS + `Serial` completed cleanly
- at that scale, CPU was faster than MPS

Mid-scale Squared comparison:

- environment: `puffer_squared`
- backend: `Serial`
- `vec.num-envs = 2`
- `env.num-envs = 1024`
- about 2048 total agents
- `train.bptt-horizon = 32`
- `train.minibatch-size = 8192`
- `train.total-timesteps = 131072`

Observed result:

- CPU finished in about `7.4s` and ended around `753K SPS`
- MPS finished in about `9.3s` and ended around `568K SPS`

Higher-scale Squared comparison:

- environment: `puffer_squared`
- backend: `Serial`
- `vec.num-envs = 8`
- `env.num-envs = 4096`
- about 32,768 total agents
- `train.bptt-horizon = 8`
- `train.minibatch-size = 32768`
- `train.total-timesteps = 524288`

Observed result:

- CPU finished in about `11.2s` and ended around `1.5M SPS`
- MPS finished in about `5.2s` and ended around `5.5M SPS`

So the local rule of thumb is:

- do not expect MPS to beat CPU on small validation jobs
- do benchmark MPS on larger workloads
- MPS only pulled clearly ahead once we got much closer to the default large
  Squared scale

Also note:

- the dashboard `GPU` and `VRAM` lines are not meaningful for MPS here
- use macOS system tools if you need real Apple GPU visibility

## Squared

### What mastery means in Squared

Squared is a tiny single-agent grid task.

The important local metric interpretation is:

- `perf` is the fraction of successful terminal episodes
- `episode_return` trends toward `+1.0` when the task is solved

For Squared, mastery means:

- `perf` near `1.0`
- `episode_return` near `+1.0`

### What worked

The standard bounded smoke path is enough to validate setup.

The default RL config, even after longer tuning attempts, plateaued well below
perfect play locally. Treat that as a known practical limitation of the
default local Squared training setup in this checkout, not as evidence that
the task is hard. For a guaranteed "watch it solve the task" result, we
trained the standard recurrent Squared policy by imitation on the optimal move
rule and saved the mastered checkpoint locally.

Current mastered Squared checkpoint:

- `experiments/puffer_squared_xxx.pt`

Current verified behavior:

- greedy success `0.99999` over 100,000 episodes
- sampled success `0.99606` over 100,000 episodes

### Watch the mastered Squared agent

```bash
cd /pufferlib/PufferLib
source .venv/bin/activate
puffer eval puffer_squared --load-model-path latest --train.device cpu
```

This opens the render window and keeps running until interrupted.

## Breakout: what we learned

### What mastery means in Breakout

Breakout logs:

- `score`
- `episode_return`
- `perf`

For Breakout, `perf` is normalized score:

```text
perf = score / max_score
```

For the default local Breakout config:

- `max_score = 864`

So "mastery" in practice means:

- `perf` approaching `1.0`
- `score` approaching `864`

### Which policy to use

Use the stock Breakout config unless you have a reason not to.

That means:

- policy family: Ocean `Policy`
- recurrent wrapper: Ocean `Recurrent`
- hidden size: `128`

This is the config already defined for `puffer_breakout`, and it trained
cleanly locally.

### Breakout smoke tests

These were verified locally:

- `puffer eval puffer_breakout --train.device cpu`
- short CPU training smoke
- short MPS training smoke

The default large-agent Breakout setup ran cleanly on both CPU and MPS.

On the short smoke we used for health/performance checking:

- MPS ended slightly faster than CPU on this machine
- so MPS is the recommended device for long Breakout training here

### Best current Breakout training recipe

If you are starting a new long Breakout run from the last clean top-level
checkpoint, use:

```bash
cd /pufferlib/PufferLib
source .venv/bin/activate
puffer train puffer_breakout \
  --tag breakout-mastery-resume-mps \
  --load-model-path latest \
  --train.device mps \
  --vec.backend Multiprocessing \
  --vec.num-envs 8 \
  --env.num-envs 4096 \
  --train.total-timesteps 2000000000 \
  --train.checkpoint-interval 250
```

If you are resuming the current interrupted local mastery attempt, use the
explicit intermediate checkpoint instead:

```bash
cd pufferlib/PufferLib
source .venv/bin/activate
puffer train puffer_breakout \
  --tag breakout-mastery-resume-mps \
  --load-model-path pufferlib/PufferLib/experiments/puffer_breakout_177353608103/model_puffer_breakout_000954.pt \
  --train.device mps \
  --vec.backend Multiprocessing \
  --vec.num-envs 8 \
  --env.num-envs 4096 \
  --train.total-timesteps 2000000000 \
  --train.checkpoint-interval 250
```

Why this is the recommended recipe:

- it uses the stock Breakout model family
- it uses MPS, which was locally healthy and slightly faster in the smoke
- it pushes `env.num-envs` high enough to keep the workload large
- it keeps `vec.num-envs = 8`, which was already stable
- it spaces checkpoints out because checkpoint interval is epoch-based

### Latest Breakout result

The most recent long run using the large-agent MPS recipe finished cleanly
and solved Breakout locally.

Final result:

- live dashboard reached `perf = 1.000`
- the finished top-level checkpoint is:

```text
experiments/puffer_breakout_177353608103.pt
```

- the run directory also contains periodic nested checkpoints up through:

```text
experiments/puffer_breakout_177353608103/model_puffer_breakout_000954.pt
```

### Watch the solved Breakout agent

```bash
cd pufferlib/PufferLib
source .venv/bin/activate
puffer eval puffer_breakout \
  --load-model-path /pufferlib/PufferLib/experiments/puffer_breakout_xxx.pt \
  --train.device cpu
```

`latest` also works now because the run finished cleanly and wrote the top-level
checkpoint.

## Parallelism guidance

If you appear to have spare CPU/GPU capacity, do not automatically keep
raising both parallelism knobs forever.

Why not:

- more parallelism changes the optimization regime, not just throughput
- with `batch_size = auto`, more total agents means much larger rollout
  batches per update
- if `train.total-timesteps` stays fixed, you get fewer update cycles
- in the simplest case, doubling batch size at fixed total timesteps gives you
  about half as many parameter updates
- larger batches can increase `SPS` while making learning per minute worse
- `vec.num-envs` adds extra multiprocessing overhead and Python process cost

The practical rule:

1. Raise `env.num-envs` first
2. Only raise `vec.num-envs` after `env.num-envs` stops helping
3. Judge by `perf` improvement per minute, not just `SPS`
4. If you double total agents, strongly consider increasing
   `train.total-timesteps` too

For this local Breakout setup, longer training was the right next move before
adding even more parallelism.

## Checkpoints, `latest`, and stopping runs

This is important.

### Where checkpoints go

Periodic checkpoints go inside a run directory:

```text
experiments/<env>_<run_id>/model_<env>_<epoch>.pt
experiments/<env>_<run_id>/trainer_state.pt
```

On a clean training finish, the trainer also copies the final checkpoint to a
top-level file:

```text
experiments/<env>_<run_id>.pt
```

### What `--load-model-path latest` actually finds

`latest` only looks for top-level files matching:

```text
experiments/<env>*.pt
```

That means:

- it finds the top-level copied checkpoint from a cleanly finished run
- it does not discover intermediate `model_<env>_<epoch>.pt` files nested
  inside the run directory

### Why `Ctrl+C` is not graceful here

The trainer installs a SIGINT handler that hard-exits the process.

This is a trainer defect, not a desirable shutdown behavior.

Practical consequence:

- `Ctrl+C` does not run the normal clean `close()` path
- you should not expect a fresh final top-level checkpoint to appear when you
  interrupt training

### What to do if you stop a run early

If you stop a long run before it finishes:

- use the newest saved `model_<env>_<epoch>.pt` inside the run directory
- do not assume `--load-model-path latest` points to that intermediate save

Example:

```bash
puffer train puffer_breakout \
  --load-model-path /pufferlib/PufferLib/experiments/puffer_breakout_xxx/model_puffer_breakout_xxx.pt \
  --train.device mps \
  ...
```

### Why sparse checkpointing matters at large scale

Checkpoint interval is counted in epochs, not timesteps.

At high agent counts:

- each epoch moves a lot of experience
- checkpointing too often adds unnecessary I/O and overhead

That is why the long Breakout recipe above uses:

```bash
--train.checkpoint-interval 250
```

### Optional add: metric-based early stop 

add to make_parser()
'''python
parser.add_argument('--stop-threshold', type=float, default=None,
        help='Threshold used with --stop-metric for early stopping')
'''

The trainer then supports an optional stop threshold at the CLI level:

```bash
--stop-metric environment/perf --stop-threshold 0.99
```

This stops training cleanly through the normal save path once the named logged
metric reaches the threshold. It is generic, so it can be used with any
environment metric that actually appears in the trainer logs. If needed, use
`--stop-mode lte` instead of the default `gte`.

## Logging and plotting

By default, the local CLI path gives you:

- the live dashboard
- checkpoints

What it does not reliably give you is a local persisted per-epoch metrics file
for plotting after the run. The default `NoLogger` drops those logs.

Practical consequence:

- for the current default local run style, you should treat the dashboard as
  live monitoring only and assume the curve is gone once the process exits
- if a run matters, enable persistent logging before you start it
- the simplest acceptable fix is `--wandb`, `--neptune`, or a local JSON logger

## Common beginner mistakes

### Running outside the repo venv

If you do not activate `PufferLib/.venv`, you may not have the right
dependencies or the `puffer` console script on `PATH`.

### Starting with multiprocessing

If your first run hangs or misbehaves on macOS, switch to:

```bash
--vec.backend Serial
```

Then add multiprocessing back later.

### Changing backend and device at the same time

Do not debug:

- `Serial` vs `Multiprocessing`
- CPU vs MPS

at the same time.

### Jumping to MPS too early

If CPU is not stable yet, MPS adds another variable.

On this machine, MPS only showed a clear speed advantage once the workload got
much larger.

### Expecting `eval` to terminate

For the environments we tested, `eval` is interactive and keeps running until
interrupted.

### Expecting the dashboard GPU line to prove MPS usage

It will stay at `0` for MPS here. That is expected.

### Assuming `latest` means "most recent intermediate checkpoint"

It does not. It means the newest matching top-level checkpoint file.

## Repo landmarks

These are the files worth reading once the basics work:

- `pyproject.toml`
  - defines the `puffer` console entrypoint
- `pufferlib/pufferl.py`
  - CLI, training, eval, checkpointing, dashboard logic
- `pufferlib/vector.py`
  - vector backends such as `Serial` and `Multiprocessing`
- `pufferlib/config/default.ini`
  - default global settings
- `pufferlib/config/ocean/squared.ini`
  - Squared-specific settings
- `pufferlib/config/ocean/breakout.ini`
  - Breakout-specific settings
- `pufferlib/ocean/environment.py`
  - Ocean environment registry
- `pufferlib/ocean/pysquared/pysquared.py`
  - readable Python reference version of Squared
- `pufferlib/ocean/squared/`
  - native Squared implementation
- `pufferlib/ocean/breakout/`
  - native Breakout implementation

## Day-one checklist

Use this if you want the fastest reliable first session:

```bash
cd /pufferlib/PufferLib
source .venv/bin/activate
which puffer
puffer eval puffer_breakout --train.device cpu

puffer train puffer_squared \
  --vec.backend Serial \
  --vec.num-envs 2 \
  --vec.num-workers 2 \
  --vec.batch-size 2 \
  --env.num-envs 16 \
  --train.bptt-horizon 32 \
  --train.minibatch-size 1024 \
  --train.total-timesteps 2048 \
  --train.device cpu

puffer eval puffer_squared --load-model-path latest --train.device cpu

puffer train puffer_squared \
  --vec.backend Multiprocessing \
  --vec.num-envs 2 \
  --vec.num-workers 2 \
  --vec.batch-size 2 \
  --env.num-envs 16 \
  --train.bptt-horizon 32 \
  --train.minibatch-size 1024 \
  --train.total-timesteps 2048 \
  --train.device cpu

puffer train puffer_squared \
  --vec.backend Serial \
  --vec.num-envs 2 \
  --vec.num-workers 2 \
  --vec.batch-size 2 \
  --env.num-envs 16 \
  --train.bptt-horizon 32 \
  --train.minibatch-size 1024 \
  --train.total-timesteps 2048 \
  --train.device mps
```

If all of that works, the local install and basic training path are healthy.

## Final recommendation

Treat PufferLib as a fast, performance-minded RL toolkit rather than a
teaching framework.

That mindset leads to the right local habits:

- start with a small built-in environment
- use CPU + `Serial` first
- benchmark MPS after CPU is already working
- change one thing at a time
- judge large runs by task metrics, not just `SPS`
- let long runs finish cleanly if you want the top-level final checkpoint

If you follow that order, this repo becomes much easier to reason about.
