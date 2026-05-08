# tmux Workflow

Use `tmux` to keep training jobs alive after disconnecting from SSH.

## Start a session

```bash
tmux new -s dsrl
```

Inside the session, activate the environment and start the run:

```bash
cd /local/kiten/dsrl_pi0
conda activate dsrl_pi0
python examples/train_sim.py
```

## Detach without stopping the job

Press:

```text
Ctrl-b d
```

This leaves the job running in the background.

## Reattach later

```bash
tmux attach -t dsrl
```

or, shorter:

```bash
tmux a -t dsrl
```

## List running sessions

```bash
tmux ls
```

## Kill a session

Only do this when the job should stop:

```bash
tmux kill-session -t dsrl
```

## Useful tips

Create one session per experiment:

```bash
tmux new -s libero_run_1
tmux new -s pi0_debug
```

Inside tmux, scroll with:

```text
Ctrl-b [
```

Then use arrow keys / PageUp / PageDown. Press `q` to exit scroll mode.

If the SSH connection drops, reconnect to the server and reattach:

```bash
ssh monotone
conda activate dsrl_pi0
tmux a -t dsrl
```

One tmux session with two panes:

```bash
tmux new -s dsrl
```

Inside tmux:

```bash
Ctrl-b %
```

Then run training in one pane and:
```bash
watch -n 1 nvidia-smi
```

in the other pane.
