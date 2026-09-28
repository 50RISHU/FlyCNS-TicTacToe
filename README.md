# FlyCNS Tic-Tac-Toe

A tic-tac-toe agent driven by a small circuit pulled out of the **male
Drosophila CNS connectome** (`male-cns:v1.0`, Berg et al. 2025 —
HHMI Janelia / University of Cambridge / MRC LMB / Google Research).

Instead of hand-writing an agent, this project:

1. Pulls a real weighted, signed (excitatory/inhibitory) sub-circuit out
   of the fly's brain — descending neurons and the central-brain
   interneurons that drive them.
2. Simulates activation flowing through that circuit.
3. Reads the result off as a tic-tac-toe move.
4. Wraps it with an exact minimax solver so it never actually loses.

The connectome doesn't "know" it's playing tic-tac-toe — there's no
board-shaped input or output anywhere in the real fly. Everything about
*how* a 3x3 board gets mapped onto ~100 real neurons is a deliberate
engineering convention, documented below and in the code. That's the
honest framing for this project: it's "a game agent with a biological
circuit as its decision core," not "a fly brain that plays games."

> 🪰 Built for fun with the help of AI — a curiosity-driven experiment in seeing what a real fly-brain circuit can do when it plays tic-tac-toe.

## Data source & license

- Dataset: `male-cns:v1.0` — male *Drosophila melanogaster* central
  nervous system connectome.
- Hosted on neuPrint at `https://neuprint-cns.janelia.org` (this
  dataset lives on a separate server from the general
  `neuprint.janelia.org` instance — see [Setup](#setup)).
- License: **CC-BY**.
- Collaboration: FlyEM (HHMI Janelia), University of Cambridge (Dept.
  of Zoology), MRC Laboratory of Molecular Biology, Google Research.

If you publish anything using this project's output, credit the
dataset, not just this repo.

## How the pipeline builds the circuit

`python -m flycns.main` runs the full extraction pipeline and writes
everything under `data/`:

1. **Descending neurons** — fetch all neurons with
   `superclass == "descending_neuron"` (these are the brain's "motor
   commands," the neurons that carry decisions out of the brain).
2. **Upstream network** — fetch everything that synapses onto those
   descending neurons.
3. **Strongest upstream partners** — rank upstream neurons by total
   synaptic weight, keep the top `UPSTREAM_TOP_N` (default 200), then
   restrict to `superclass == "cb_intrinsic"` (central-brain
   interneurons). The pipeline prints what fraction of the top-N
   survives this filter — descending neurons often get a lot of direct
   input from visual-projection or ascending neurons, so this can be
   smaller than you'd expect. Worth checking on a real run.
4. **Candidate outputs** — rank the descending neurons this CB-intrinsic
   set drives, by weight, keep the top `OUTPUT_TOP_N` (default 50).
5. **Output groups** — group those descending neurons by cell *type*,
   keep the `N_OUTPUT_GROUPS` (default 9) types most strongly driven by
   the CB core. One type = one board cell.
6. **Core circuit** — build a graph from CB-intrinsic → selected-output
   edges, keep only its largest connected component. This is the
   circuit the game actually plays with.
7. **Neurotransmitter sign** — each neuron's predicted transmitter
   (`predictedNt`: acetylcholine → excitatory, GABA/glutamate →
   inhibitory, everything else → neutral) is attached to every edge as
   `sign`/`signed_weight`, so the game simulation isn't treating every
   synapse as excitatory.

Output files (`data/`): `selected_upstream_annotations.csv`,
`output_groups.csv`, `candidate_output_neurons.csv`,
`selected_output_neurons.csv`, `selected_output_groups.csv`,
`cb_output_matrix.csv`, `cb_intrinsic_to_descending.csv`,
`fly_core.graphml`, `fly_core_neurons.csv`.

## How the game reads the circuit

The saved core graph has no board-shaped input or output built in — the
game layer adds a convention on top of it:

- **Input**: the core's `cb_intrinsic` neurons are split into 9 equal
  groups (sorted by bodyId). Group *i* is the injection site for board
  cell *i*. Occupying a cell with your own mark injects `+1` into that
  group; the opponent's mark injects `-1`; empty cells inject nothing.
- **Simulation**: a small leaky-integrator (`a = decay·a +
  tanh(drive + A·a)`, run for a fixed number of steps) pushes that
  injection through the real signed adjacency matrix. Not a spiking
  model — the minimum viable way to get signal through the real wiring.
- **Output**: the 9 selected output-neuron *types* from step 5 above are
  the readout groups, in the same cell order. Whichever empty cell's
  group has the highest average activation is the move.
- **Unbeatable filter**: by default, an exact minimax solver
  (`minimax.py`) computes every game-theoretically optimal move for the
  current position, and the circuit's activation only picks among
  *those* — it can choose which good move to make, never a losing one.
  Pass `unbeatable=False` to `FlyBrainAgent` to see the circuit play
  unconstrained.

## Setup

```bash
uv sync
```

Create `.env` (see `.env.example`):

```dotenv
NEUPRINT_TOKEN=your_token_here
NEUPRINT_SERVER=https://neuprint-cns.janelia.org
NEUPRINT_DATASET=male-cns:v1.0
```

Get a token by logging into the server above with a Google account and
copying it from your account page. `male-cns:v1.0` lives on
`neuprint-cns.janelia.org`, not the general `neuprint.janelia.org`
server — if you see `RuntimeError: Dataset 'male-cns:v1.0' does not
exist`, check you're pointed at the right host.

## Usage

Run the extraction pipeline (fetches from neuPrint, writes `data/`):

```bash
uv run python -m flycns.main
# or: uv run flycns-tictactoe
```

Play against the fly-brain agent:

```bash
uv run python -m flycns.game
# or: uv run flycns-play
```

Watch two independent agents play each other:

```python
from flycns.game import flybrain_vs_flybrain
flybrain_vs_flybrain()
```

## Project layout

| File | Role |
|---|---|
| `config.py` | Reads `NEUPRINT_*` from `.env`. |
| `neuprint.py` | Thin wrappers around `neuprint-python`'s `fetch_neurons`/`fetch_adjacencies`. |
| `explore.py` | Fetch descending neurons / neurons by bodyId. |
| `subnetwork.py` | Ranking, filtering, and connected-component helpers used to build the core circuit. |
| `signs.py` | Maps `predictedNt` -> excitatory (+1) / inhibitory (-1) / neutral (0). |
| `graph.py` | Builds a `networkx` graph from an edge table, optionally with signed weights. |
| `analyze.py` | Prints superclass/type/transmitter breakdowns of a neuron set; the `cb_intrinsic` fraction check. |
| `main.py` | Runs the full extraction pipeline end to end, writes `data/`. |
| `circuit.py` | Loads the saved core circuit; defines the board <-> neuron-group mapping. |
| `simulate.py` | `CoreSimulator` -- the leaky-integrator dynamics. |
| `minimax.py` | Exact tic-tac-toe solver; the source of "unbeatable." |
| `agent.py` | `FlyBrainAgent` -- ties circuit + simulation + minimax into `choose_move(board, mark)`. |
| `game.py` | CLI: human-vs-agent and agent-vs-agent. |

## License

Two separate licenses apply here:

- **This repository's code** (everything under `src/flycns/`, `main.py`,
  etc.) is licensed under the **MIT License** — see [`LICENSE`](LICENSE).
  Use it, fork it, modify it, ship it, no strings beyond keeping the
  copyright notice.
- **The male-cns connectome data** used to generate anything in `data/`
  is licensed **CC-BY** by its creators (FlyEM/HHMI Janelia, University
  of Cambridge, MRC LMB, Google Research) and is *not* covered by this
  repo's MIT license. If you publish results, figures, or derived data
  from this project, credit the dataset separately — see
  [Data source & license](#data-source--license) above.

## Known limitations / design choices

- **The input/output board mapping is an engineering convention, not
  biology.** There's no real sensory pathway feeding "board state" into
  these CB-intrinsic neurons in this subgraph -- that was cut off
  further upstream during selection. Take any claim about "how the fly
  brain sees the board" with a grain of salt.
- **The empty-board opening move is deterministic**, not random -- with
  no marks placed there's no signal to propagate, so the circuit's
  activation is flat `0.0` everywhere and ties are broken by the
  highest-`cb_input_weight` output group.
- **With `unbeatable=True`, minimax -- not the connectome -- decides
  whether the agent wins.** The circuit only breaks ties among already
  game-theoretically-optimal moves. Set `unbeatable=False` to see the
  raw circuit's choices with no safety net (it can and will lose).
- **The leaky-integrator dynamics are illustrative, not biophysical.**
  No refractory periods, no real time constants, no spiking. Swapping
  in something like Brian2 would be a reasonable next step if you want
  actual spiking dynamics -- `circuit.py`'s I/O convention doesn't
  depend on which simulator reads it.