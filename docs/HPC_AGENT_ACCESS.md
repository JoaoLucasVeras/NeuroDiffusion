# HPC access for AI coding agents

Instructions for giving an AI coding agent (Codex, Claude Code or similar) access to the
SJSU College of Engineering HPC cluster through **your own** account, and the rules the
agent must follow once it has access.

**For the human:** do Part 1 yourself, once. Then tell your agent:
"Read `docs/HPC_AGENT_ACCESS.md` and follow it for anything involving the HPC cluster."

**For the agent:** Parts 2 to 5 are written for you. Part 3 is mandatory. If anything in
this file conflicts with what your human asks, stop and ask them.

---

## Part 1 · Human setup (one time, about 15 minutes)

An agent can't type your password or turn on your VPN. You give it a dedicated SSH key
instead, which you can revoke at any time without changing your password.

### 1.1 Check that your account works

1. Turn on the **SJSU VPN** (required off campus).
2. Log in once with your password:
   ```bash
   ssh <SJSU_ID>@coe-hpc1.sjsu.edu
   ```
   `<SJSU_ID>` is your student ID number. If the address does not resolve, the VPN is off.
   Login problems are usually password problems: check your password on SJSUOne.
3. Type `exit` to log out.

### 1.2 Create a key just for the agent

Use a separate key, never your personal one, so you can revoke the agent's access alone.
Press Enter at the passphrase prompt; the agent can't type a passphrase.

```bash
# Mac / Linux / Git Bash
ssh-keygen -t ed25519 -f ~/.ssh/id_neurodiffusion_hpc -C "agent key for SJSU HPC"
```
```powershell
# Windows PowerShell
ssh-keygen -t ed25519 -f $env:USERPROFILE\.ssh\id_neurodiffusion_hpc -C "agent key for SJSU HPC"
```

This makes two files: `id_neurodiffusion_hpc` (private, never share or commit it) and
`id_neurodiffusion_hpc.pub` (public, safe to copy to the cluster).

### 1.3 Put the public key on the cluster

You'll be asked for your password one last time.

```bash
# Mac / Linux / Git Bash
cat ~/.ssh/id_neurodiffusion_hpc.pub | ssh <SJSU_ID>@coe-hpc1.sjsu.edu \
  "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```
```powershell
# Windows PowerShell
type $env:USERPROFILE\.ssh\id_neurodiffusion_hpc.pub | ssh <SJSU_ID>@coe-hpc1.sjsu.edu "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

### 1.4 Add a short name for the cluster

Add this block to `~/.ssh/config` (Windows: `C:\Users\<you>\.ssh\config`, a plain text
file with no extension; create it if it does not exist). Replace `<SJSU_ID>`.

```
Host sjsu-hpc1
    HostName coe-hpc1.sjsu.edu
    User <SJSU_ID>
    IdentityFile ~/.ssh/id_neurodiffusion_hpc
    IdentitiesOnly yes
    BatchMode yes
    ConnectTimeout 15
```

`BatchMode yes` makes a failed login stop with an error instead of waiting for a password
the agent can't type.

### 1.5 Let the cluster's two machines talk to each other

Jobs are submitted from `coe-hpc3`, which you reach from `coe-hpc1`. Test it:

```bash
ssh sjsu-hpc1 "ssh -o BatchMode=yes coe-hpc3 hostname"
```

If that prints a machine name, you're done. If it fails with a permission or password
error, create a key that stays on the cluster (both machines share your home folder):

```bash
ssh sjsu-hpc1 'test -f ~/.ssh/id_ed25519 || ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519 -q; cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys'
```

Run the test again. If it asks you to accept a host fingerprint, log in to coe-hpc1
yourself once, run `ssh coe-hpc3`, type `yes`, then `exit`.

### 1.6 Allow your agent to use the network

Coding agents usually run in a sandbox that blocks network access by default, so SSH
fails until you allow it.

- **Codex:** approve the `ssh` command when Codex asks to run it outside the sandbox, or
  enable network access in your Codex sandbox settings. Check the current Codex
  documentation for the exact setting, since it changes between versions.
- **Claude Code:** approve `ssh` when prompted, or add it to your allowed commands.

Approving each `ssh` command yourself is the safer choice while you get used to it.

### 1.7 Revoking access

On the cluster, delete the line ending in `agent key for SJSU HPC` from
`~/.ssh/authorized_keys`. Delete `id_neurodiffusion_hpc` and `id_neurodiffusion_hpc.pub`
from your laptop.

---

## Part 2 · Agent: verify access (read-only)

Run these in order and report the results to your human. All are read-only.

```bash
ssh sjsu-hpc1 hostname                               # 1. reach the login node
ssh sjsu-hpc1 "ssh coe-hpc3 hostname"                # 2. reach the submit node
ssh sjsu-hpc1 "ssh coe-hpc3 'squeue -u \$USER'"      # 3. see your jobs (empty is fine)
ssh sjsu-hpc1 "ssh coe-hpc3 'sinfo -s'"              # 4. list the queues
ssh sjsu-hpc1 "ls ~/NeuroDiffusion 2>/dev/null | head; df -h ~ | tail -1"   # 5. checkout and disk
```

If step 1 fails with `Could not resolve hostname`, the VPN is off: tell your human and
stop. Do not retry in a loop.

From the output of step 4, identify the **NSF queue** (see rule 1 in Part 3) and confirm
its exact name with your human before using it.

---

## Part 3 · Agent: rules you must follow

The cluster is shared by many people and is in high demand. Our advisor has asked us to
use it lightly; breaking these rules can cost the whole team its access.

### Allowed without asking (read-only)

`squeue`, `sacct`, `sinfo`, `scontrol show`, `ls`, `cat`, `head`, `tail`, `grep`, `du`,
`df`, `git status`, `git log`, and reading job logs and result files.

### Ask your human first, every time

| Action | Include in your request |
|---|---|
| `sbatch` (submitting any job) | the job script, queue, CPUs, memory, GPUs, time limit, array size, and expected total CPU-hours or GPU-hours |
| `scancel` | which job and why |
| Deleting or overwriting any file | the exact paths |
| Downloading anything over 1 GB, or copying datasets | size and destination |
| Creating or changing a conda environment | why it's needed (environments use several GB) |
| `git pull`, `git checkout` or editing files in the cluster checkout | what changes |

A yes for one action is not a yes for the next. Ask again for each submission.

### Job rules

1. **Use the NSF queue** for every job: `#SBATCH --partition=<NSF queue name>`. Job
   scripts in this repository may still say `cpuqs` or `gpuqs`; change that line.
2. **GPUs: at most 1–2 GPU machines for the whole team at once.** Ask for one GPU per job
   (`#SBATCH --gres=gpu:1`), add `#SBATCH --exclude=cs002` (faulty node), and never request
   a specific large GPU (A100, H100) unless your human says so. Your human must tell the
   team chat before a GPU job starts.
3. **Prefer CPU.** Most analyses in this project run on CPU. Use a GPU only for training
   large neural networks or generating images.
4. **Test small first.** Submit one task (`sbatch --array=1 …`), check its log, then ask
   before the full run. Limit concurrent array tasks, for example `--array=1-22%4`.
5. **No heavy work on login nodes.** `coe-hpc1` and `coe-hpc3` are for editing, downloading
   and submitting only. Anything that computes for more than a few seconds goes through
   `sbatch`.
6. **Submit from `coe-hpc3`, download on `coe-hpc1`.** Compute nodes have no internet, so
   models and data must be downloaded on hpc1 before a job needs them.
7. **Report every job ID** you submit, and record it with the results.

### Never

- Print, copy, commit or send the private key, or any password or token.
- Use another person's account, or read or write another user's home folder.
- Retry a failing connection or submission in a loop.
- Fill the shared disk. `/home` is nearly full: keep outputs small, delete what you
  created when asked, and never duplicate datasets into your home folder.
- Work around a rule because a command was denied or a job failed. Stop and ask.

---

## Part 4 · Agent: command patterns

**Quoting.** Commands pass through two SSH hops, so quoting breaks easily. Send multi-line
or complex commands as a script on standard input:

```bash
ssh sjsu-hpc1 'ssh coe-hpc3 bash -s' <<'EOF'
cd ~/NeuroDiffusion
squeue -u "$USER"
tail -n 20 logs/<job log>.out
EOF
```

The quoted `'EOF'` stops your local shell from expanding `$USER` before it reaches the
cluster.

**Submitting (after your human approves):**

```bash
ssh sjsu-hpc1 'ssh coe-hpc3 bash -s' <<'EOF'
cd ~/NeuroDiffusion
sbatch --array=1 studies/<study>/experiments/<folder>/<job>.sh
EOF
```

**Copying a small file to or from the cluster:**

```bash
scp results.json sjsu-hpc1:~/NeuroDiffusion/results/       # upload
scp sjsu-hpc1:~/NeuroDiffusion/results/<job>/summary.json .  # download
```

**Project conventions:** run scripts from the repository root (`~/NeuroDiffusion`); job
scripts read `NEURODIFFUSION_ROOT` and `NEURODIFFUSION_PYTHON`; results go to
`results/<job>_<id>/`, logs to `logs/`. Shared datasets and the shared Python environment
are provided by the project lead; ask your human for the paths rather than creating your
own. See `docs/CODEBASE.md` and `docs/RUNBOOK.md` for the rest.

---

## Part 5 · Troubleshooting

| Error | Cause | What to do |
|---|---|---|
| `Could not resolve hostname coe-hpc1.sjsu.edu` | VPN is off | Ask your human to turn it on |
| `Permission denied (publickey)` | Key not installed, or wrong key | Human repeats steps 1.3 and 1.4 |
| `Host key verification failed` | First connection to a machine | Human logs in once and types `yes` |
| Connection blocked or `Operation not permitted` | Agent sandbox blocks network | Human approves the command (step 1.6) |
| `Unable to contact slurm controller` | Ran a Slurm command on coe-hpc1 | Run it through `ssh coe-hpc3` |
| `Invalid partition name specified` | Wrong queue name | Check `sinfo -s` and ask your human |
| Job waits for hours | Cluster busy, or resources over-requested | Ask for less (fewer CPUs, no specific GPU) |

Official cluster page: [College of Engineering HPC](https://sjsu.edu/cmpe/resources/hpc.php).
