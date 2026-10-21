# BioQuantumFly — Environment

Reference for the development and simulation environment on the research
server `hayate`. Everything described here is isolated, reproducible, and
stored under `/home`.

---

## 1. System

| Item | Value |
|---|---|
| OS | Ubuntu 24.04.4 LTS (Noble Numbat) |
| Kernel | Linux 6.8.0-136-generic |
| Architecture | x86_64 |
| Host | `hayate` |
| Logical CPUs | 384 |
| RAM | ~1.5 TiB |
| Conda | 25.5.1 (base: `/opt/miniconda`, read-only to us) |
| Git | 2.43.0 |
| C compiler | gcc 13.3.0 (`/usr/bin/gcc`) |
| C++ compiler | g++ 13.3.0 (`/usr/bin/g++`) |
| System Python | 3.13.5 — **not modified, not used by this project** |

## 2. Conda environment

| Item | Value |
|---|---|
| Prefix | `~/conda-envs/bioquantumfly` (`/home/sohma/conda-envs/bioquantumfly`) |
| Python | 3.12.14 |
| Interpreter | `~/conda-envs/bioquantumfly/bin/python` |

The environment is a **path-based** (`-p`) environment, deliberately *not* a
named one, so that it lives under `/home` rather than in a conda-managed
directory. The conda base install at `/opt/miniconda` is not writable by us,
so package downloads fall through to `~/.conda/pkgs`; we also set
`CONDA_PKGS_DIRS` explicitly to make that guarantee rather than a side effect.

The base environment is never used for project dependencies.

## 3. Installed packages

Conda supplies only the interpreter and `pip`. Every scientific package comes
from pip, so a single resolver owns the stack and there is no conda/pip ABI
mixing around numpy.

| Package | Version | Role |
|---|---|---|
| numpy | 2.5.3 | arrays / linear algebra |
| scipy | 1.18.1 | ODEs, sparse linear algebra |
| pandas | 3.0.5 | connectome tables |
| pyarrow | 25.0.1 | Parquet I/O for cached subgraphs |
| networkx | 3.6.1 | MaleCNS graph processing |
| matplotlib | 3.11.1 | figures |
| sympy | 1.14.0 | symbolic Hamiltonians |
| cython | 3.1.3 | Brian2 code generation |
| jinja2 | 3.1.6 | Brian2 templating |
| **qutip** | **5.3.1** | density matrices, Lindblad dynamics |
| **brian2** | **2.10.1** | LIF spiking neural simulation |
| **neuprint-python** | **0.6.3** | MaleCNS connectome access |
| pyyaml | 6.0.3 | configs |
| pydantic | 2.13.5 | config validation |
| typer | 0.27.2 | CLI |
| rich | 15.0.0 | terminal output |
| tqdm | 4.70.0 | progress bars |
| pytest / pytest-cov | 9.1.1 / 7.1.0 | testing |

Deliberately **not** installed: `qiskit`, `torch`, `tensorflow`, `jax`,
`cupy`, CUDA. The initial model needs only density matrices, Lindblad
dynamics, radical-pair physics, LIF simulation, and graph processing — QuTiP
and Brian2 cover all of it.

## 4. Filesystem notes — important

The root filesystem is nearly full and must be kept clear:

| Mount | Size | Free | Use |
|---|---|---|---|
| `/` | 877 G | ~52 G (94 % used) | **avoid** |
| `/home` (NFS, `203.178.162.135:/volume1/only14TB`) | 14 T | ~10.9 T | everything |

Resolved paths, all on `/home`:

- Project: `/home/sohma/aqua/Brain-of-Drosophila`
- Conda env: `/home/sohma/conda-envs/bioquantumfly`
- Conda package cache: `/home/sohma/.conda/pkgs`
- Pip cache: `/home/sohma/.cache/pip`
- Scratch: `$BQF_ROOT/data/cache/tmp` (via `TMPDIR`)

`scripts/activate_env.sh` points `TMPDIR` into the project tree, because the
default `/tmp/user/$UID` is root-backed. Downloaded MaleCNS data goes to
`data/raw/`, caches to `data/cache/`, simulation output to `results/` — all
under the project directory, all git-ignored.

`scripts/check_environment.py` fails if the project or the environment is
found on `/`, so this cannot regress silently.

Note that `/home` is NFS-mounted. Many-small-file workloads (conda operations,
Brian2 C++ standalone builds) are slower than on local disk. This is a
throughput trade-off accepted in exchange for capacity.

## 5. CPU thread policy

The machine has 384 logical CPUs and is shared. Numerical libraries default
to grabbing every core, which is antisocial and usually slower for our problem
sizes. The default is therefore **16 threads**:

```
OMP_NUM_THREADS=16
OPENBLAS_NUM_THREADS=16
MKL_NUM_THREADS=16
NUMEXPR_NUM_THREADS=16
VECLIB_MAXIMUM_THREADS=16
```

Set by `scripts/activate_env.sh`; override with `BQF_THREADS=32 source
scripts/activate_env.sh`. The test suite pins 4 threads in
`tests/conftest.py`. No shell startup file has been modified.

## 6. Activating the environment

```bash
cd ~/aqua/Brain-of-Drosophila
source scripts/activate_env.sh
```

This runs the real conda activation hook (so `CONDA_PREFIX` and any
`activate.d` scripts are honoured), applies the thread limits, and redirects
`TMPDIR`. It prints the interpreter path and whether a neuPrint token is
present.

Verify:

```bash
python --version     # Python 3.12.14
which python         # /home/sohma/conda-envs/bioquantumfly/bin/python
```

## 7. Running the checks

```bash
python scripts/check_environment.py   # full validation, one command
pytest -q                             # test suite
```

`check_environment.py` verifies the Python version and interpreter location,
all core imports, a real QuTiP Lindblad evolution, a real Brian2 simulation,
compiler presence, filesystem placement, thread policy, and neuPrint
availability. Exit code 0 means usable. A missing `NEUPRINT_TOKEN` is a
warning, never a failure.

The test suite requires **no internet access and no token**.

## 8. Configuring `NEUPRINT_TOKEN`

The connectome client needs a personal token from
<https://neuprint.janelia.org> (Account → Auth Token).

```bash
export NEUPRINT_TOKEN="<your token>"
```

Put it in a shell file outside the repository (e.g. `~/.neuprint_token`,
`chmod 600`) and source it, or paste it per session.

**Security rules:**

- Never commit a token. `.env`, `.env.*` (except `.env.example`) and `*.token`
  are git-ignored.
- Never print, echo, or log a token value. Tooling here reports only presence.
- Never hard-code a token in source. `tests/test_neuprint_smoke.py` scans the
  tracked source tree for JWT-shaped literals and fails if it finds one.
- If a token is ever committed, revoke it at Janelia immediately — removing
  the commit is not sufficient.

## 9. Connectome dataset (not yet downloaded)

The full MaleCNS dataset is intentionally **not** fetched during setup. Later
work will use:

```python
server  = "https://neuprint.janelia.org"
dataset = "male-cns:v1.0"
```

starting from small subgraphs. Anything downloaded must land in `data/raw/`
under `/home`.

## 10. Reproducing this environment

```bash
export CONDA_PKGS_DIRS=$HOME/.conda/pkgs
mkdir -p ~/conda-envs
conda create -p ~/conda-envs/bioquantumfly python=3.12 pip -y
conda activate ~/conda-envs/bioquantumfly
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-lock.txt   # exact pinned versions
```

`environment.yml` records the same stack at the loose-dependency level;
`requirements-lock.txt` is the exact pin set from `pip freeze`.

## 11. Known warnings

- `NEUPRINT_TOKEN` unset — expected until connectome work begins.
- `/opt/miniconda/pkgs` is not writable, so conda logs a fallback to
  `~/.conda/pkgs`. This is the behaviour we want.
- Brian2 runs with the `numpy` codegen target in tests for determinism and to
  avoid a compile step. C++ standalone mode is available (g++ 13.3.0 detected)
  but intentionally not enabled yet.
