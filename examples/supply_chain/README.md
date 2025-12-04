# Supply Chain Task (Curriculum for ShinkaEvolve)

This example defines a **curriculum of supply-chain logistics tasks** for ShinkaEvolve.
All levels share a **unified input–output interface** for the evolved policy, but differ in
dynamics, uncertainty, and the objective function.

The goal is to study:

- how ShinkaEvolve evolves **logistics policies** under increasing task complexity, and
- how **curriculum learning on the task side** affects sample efficiency, robustness, and
  the structure of discovered algorithms.

We provide 5 difficulty levels:

1. Level 1 — Single-period, deterministic routing
2. Level 2 — Multi-period inventory dynamics
3. Level 3 — Stochastic demand
4. Level 4 — Contested logistics (edge failure & risk)
5. Level 5 — Full risk-aware, long-horizon, multi-echelon logistics

## 1. Unified Interface Across Levels

### 1.1 Evolved function

Across **all levels**, ShinkaEvolve is asked to evolve a single function:

```python
# EVOLVE-BLOCK-START
def logistics_policy(state: LogisticsState) -> Shipments:
    """Decide shipments on each edge for the current time step."""
    ...
# EVOLVE-BLOCK-END
```

The **signature is fixed** across levels:

- Input: `state: LogisticsState` — current logistics state
- Output: `shipments: Shipments` — non-negative shipment amounts per directed edge

The environment (evaluation script) is responsible for:

- applying capacity / inventory / feasibility constraints,
- simulating inventory dynamics and losses over time,
- computing scalar rewards / scores.

The curriculum only changes **how `LogisticsState` is populated** and **how rewards are computed**, not the type of the function to be evolved.

### 1.2 State representation

Conceptually, `LogisticsState` exposes the following fields to the policy:

```python
@dataclass
class LogisticsState:
    # Time
    t: int          # current time index (0-based)
    T: int          # total horizon length

    # Nodes and edges
    num_nodes: int
    edges: List[Tuple[int, int]]   # directed edges e = (i, j), indexed by e = 0..E-1

    # Node attributes at time t
    inventory: List[float]         # I_i,t ≥ 0
    demand: List[float]            # external demand d_i,t ≥ 0 (0 if no demand at node i)
    is_supply_node: List[bool]     # flags for depot / source nodes
    is_demand_node: List[bool]     # flags for demand nodes (frontline, etc.)

    # Edge attributes at time t
    edge_capacity: List[float]     # u_e ≥ 0 (per-period capacity)
    edge_cost: List[float]         # c_e ≥ 0 (shipping cost per unit)
    edge_available: List[bool]     # a_e,t ∈ {0,1} (usable or blocked)
    edge_risk: List[float]         # r_e,t ∈ [0,1] (loss probability if used)

    # Optional global parameters / hyper-parameters (may be unused in some levels)
    cost_weight: float             # γ ≥ 0 (weight for transport cost)
    risk_weight: float             # β ≥ 0 (weight for loss / risk)
    supply_reward_weight: float    # α ≥ 0 (weight for fulfilled demand)
```

**Important:**

- This structure is the **superset** needed for the most complex level (Level 5).
- Simpler levels (1–3) simply **set some fields to default values**:

  - `edge_available[:] = True`, `edge_risk[:] = 0.0` when there is no contested logistics
  - `T = 1` for single-period tasks
  - `risk_weight = 0.0` when risk is not part of the objective

The policy can safely ignore fields that are not relevant for a given level, but the type
never changes.

---

### 1.3 Action representation

`Shipments` is a length-`E` vector aligned with `state.edges`:

```python
Shipments = List[float]  # shipments[e] = x_e,t ≥ 0
```

Each component `x_e,t` represents the amount shipped along directed edge `e = (i, j)` during
the current period `t`.

Constraints (enforced by the environment):

- **Non-negativity**:
  $x\_{e,t} \ge 0 \quad \forall e$

- **Edge capacity**:
  $x\_{e,t} \le a\_{e,t} \cdot u_e \quad \forall e$,
  where (a\_{e,t} \in {0,1}) is the edge availability and (u_e) is capacity.
- **Inventory conservation** at each node (i):
  $\sum_{j:(i,j)\in E} x_{(i,j),t} \le I_{i,t}$

If any hard constraint is violated, the evaluation script either:

- clips the shipment to the feasible range, or
- assigns a very large negative score (depending on the configuration).

## 2. General Mathematical Model

We summarize the **most general version** (Level 5); lower levels will be defined as
special cases.

Let:

- $N$: set of nodes, $|N| = n$
- $E \subseteq N \times N$ — set of directed edges, $|E| = m$
- $T = \{0,1,\dots, T-1\}$ — time horizon (discrete periods)
- For scenarios ($\omega$) (used in stochastic levels), we index them by $\omega \in \Omega$ with probabilities $p^\omega$.

### 2.1 State variables

For each node $i \in N$, time $t$, scenario $\omega$:

- Inventory: $I_{i,t}^\omega \ge 0$
- Demand (exogenous): $d_{i,t}^\omega \ge 0$
- Fulfilled demand at node $i$ and time $t$: $s_{i,t}^\omega \ge 0$

For each edge $e = (i,j) \in E$, time $t$, scenario $\omega$:

- Shipment decision (action): $x_{e,t}^\omega \ge 0$
- Availability (exogenous, contested levels only): $a_{e,t}^\omega \in \{0,1\}$
- Loss indicator (random, contested levels only): $\xi_{e,t}^\omega \in \{0,1\}$ with $\mathbb{P}(\xi_{e,t}^\omega = 1) = r_{e,t}^\omega$.

### 2.2 Dynamics and constraints

**Inventory balance** at each node $i$ and time $t$:

$I_{i,t+1}^\omega = I_{i,t}^\omega + \sum_{j:(j,i)\in E} \hat{x}_{(j,i),t}^\omega - \sum_{j:(i,j)\in E} x_{(i,j),t}^\omega + s_{i,t}^\omega.$

Here $s_{i,t}^\omega$ is treated as local production/supply added at the node; if you model it as demand served (a consumption term), flip its sign accordingly.

where $\hat{x}_{e,t}^\omega$ is the **effective arrival** on edge $e$. In uncontested levels:

$\hat{x}_{e,t}^\omega = x_{e,t}^\omega$

In contested levels:

$\hat{x}_{e,t}^\omega = a_{e,t}^\omega \cdot (1 - \xi_{e,t}^\omega) \cdot x_{e,t}^\omega.$

**Demand fulfillment** is constrained by inventory:

$0 \le s_{i,t}^\omega \le d_{i,t}^\omega, \quad s_{i,t}^\omega \le I_{i,t}^\omega + \sum_{j:(j,i)\in E} \hat{x}_{(j,i),t}^\omega.$

**Capacity constraints** on edges:

$0 \le x_{e,t}^\omega \le a_{e,t}^\omega \cdot u_e.$

### 2.3 Objective (single scenario)

For a given policy $\pi$ and scenario $\omega$, we define:

- Total fulfilled demand:
  $R_{\text{sup}}(\pi,\omega) = \sum_{t \in T} \sum_{i \in N} s_{i,t}^\omega.$
- Transport cost:
  $C_{\text{trans}}(\pi,\omega) = \sum_{t \in T} \sum_{e \in E} c_e \cdot x_{e,t}^\omega.$
- Loss cost (contested levels only):
  $L_{\text{loss}}(\pi,\omega) = \sum_{t \in T} \sum_{e \in E} h_e \cdot \xi_{e,t}^\omega x_{e,t}^\omega.$

The **per-scenario score** is:

$F(\pi,\omega) = \alpha R_{\text{sup}}(\pi,\omega) - \gamma C_{\text{trans}}(\pi,\omega) - \beta L_{\text{loss}}(\pi,\omega).$

with non-negative weights (\alpha,\gamma,\beta).

### 2.4 Overall objective

- **Expectation-based** (Levels 1–4):

$J(\pi) = \mathbb{E}_{\omega \sim p}[F(\pi,\omega)] \approx \frac{1}{|\Omega_{\text{eval}}|} \sum_{\omega \in \Omega_{\text{eval}}} F(\pi,\omega).$

- **Risk-aware variant** (Level 5, optional):

Define the loss random variable $Z(\pi,\omega) = \gamma C_{\text{trans}}(\pi,\omega) - \beta L_{\text{loss}}(\pi,\omega)$.

We can use a CVaR-style objective:

$J_{\text{risk}}(\pi) = \mathbb{E}[R_{\text{sup}}(\pi,\omega)] - \lambda \, \text{CVaR}_\tau(Z(\pi,\omega)).$

with risk aversion parameter (\lambda \ge 0) and confidence level (\tau \in (0,1)).

---

## 3. Level Definitions

Each level is a **restriction / specialization** of the general model above.
The policy interface stays the same; we only change:

- the **environment dynamics / parameters**, and
- the **score** used by the evaluator.

### 3.1 Level 1 — Single-period deterministic routing

**Goal:** Learn basic “ship goods where needed without wasting cost”.

- Horizon:

  - (T = 1).

- Nodes:

  - One depot (supply node), multiple demand nodes.

- Demands:

  - Deterministic vector (d_i \ge 0) (no time dimension).

- Contested:

  - No contested effects:

    - $a_{e,0}^\omega = 1$,
    - $r_{e,0}^\omega = 0$,
    - $\hat{x}_{e,0}^\omega = x_{e,0}^\omega$.

- Dynamics:

  - No meaningful inventory dynamics (single period).

- Objective (per instance):

  - Let $\text{fulfilled} = \sum_i s_{i,0}$, $\text{cost} = \sum_e c_e x_{e,0}$.
  - Score:
    $F_{\text{Lv1}}(\pi) = \text{fulfilled} - \lambda_{\text{cost}} \cdot \text{cost},$
    with $(\lambda_{\text{cost}} > 0)$.

The environment samples multiple deterministic instances (different demand patterns) and
averages the score across them.

### 3.2 Level 2 — Multi-period inventory dynamics

**Goal:** Learn to trade off “serve now vs. save inventory for later”.

- Horizon:

  - $T > 1$ (e.g., 5–10 periods).

- Nodes:

  - Same basic graph; optionally a simple intermediate hub can be added.

- Demands:

  - Deterministic time series $(d_{i,t})$.

- Contested:

  - None: $(a_{e,t}^\omega = 1)$, $(r_{e,t}^\omega = 0)$.

- Dynamics:

  - Full inventory balance:
    $I_{i,t+1} = I_{i,t} + \sum_{j:(j,i)\in E} x_{(j,i),t} - \sum_{j:(i,j)\in E} x_{(i,j),t} - s_{i,t},$
    with the same demand fulfillment constraints as above.

- Objective:
  $F_{\text{Lv2}}(\pi) = \sum_{t} \sum_i s_{i,t} - \lambda_{\text{cost}} \sum_{t} \sum_e c_e x_{e,t}.$

This level introduces **temporal coupling**: shipping too much too early can cause
shortages later.

### 3.3 Level 3 — Stochastic demand

**Goal:** Learn policies that are robust to **demand uncertainty**.

- Horizon:

  - Same as Level 2 ($T > 1$).

- Demands:

  - For each evaluation run scenario ($\omega$), a random demand sequence
    $(d_{i,t}^\omega)$ is sampled from a fixed distribution (e.g., around some mean with noise).

- Contested:

  - None: $(a_{e,t}^\omega = 1)$, $(r_{e,t}^\omega = 0)$.

- Dynamics:

  - Same inventory balance and fulfillment constraints as Level 2.

- Objective:

  - Same shape as Level 2, but **expected** over scenarios:
    $J_{\text{Lv3}}(\pi) = \mathbb{E}_{\omega}\!\left[\sum_{t} \sum_i s_{i,t}^\omega - \lambda_{\text{cost}} \sum_{t} \sum_e c_e x_{e,t}^\omega\right].$

Evaluation is done by Monte Carlo:

- sample several scenarios ($\omega$) (different demand trajectories),
- run the same policy ($\pi$) in each,
- average the resulting scores.

---

### 3.4 Level 4 — Contested logistics (edge failure & risk)

**Goal:** Learn to be careful with **dangerous routes** and maintain robustness under
intermittent failures.

- Horizon:

  - Same as Level 3.

- Demands:

  - Can be deterministic or stochastic (typically keep Level 3’s setup).

- Contested edges:

  - Each edge ($e$) has:

    - availability $a_{e,t}^\omega \in \{0,1\}$ (blocked vs. usable),
    - loss probability $r_{e,t}^\omega \in [0,1]$.

  - If the edge is used:

    - a Bernoulli loss $\xi_{e,t}^\omega \sim \text{Ber}(r_{e,t}^\omega)$ is drawn, and
    - only $\hat{x}_{e,t}^\omega = (1 - \xi_{e,t}^\omega) x_{e,t}^\omega$ arrives at the destination.

- Dynamics:

  - Inventory uses the contested arrival $\hat{x}_{e,t}^\omega$ as in Section 2.2.

- Objective:

  - We add a loss penalty:
    $F_{\text{Lv4}}(\pi,\omega) = \sum_{t} \sum_i s_{i,t}^\omega - \lambda_{\text{cost}} \sum_{t} \sum_e c_e x_{e,t}^\omega - \lambda_{\text{loss}} \sum_{t} \sum_e h_e \, \xi_{e,t}^\omega x_{e,t}^\omega.$

  - Evaluation aggregates over scenarios as in Level 3:
    $J_{\text{Lv4}}(\pi) = \mathbb{E}_\omega[F_{\text{Lv4}}(\pi,\omega)].$

This level encourages policies that:

- avoid overly risky edges,
- maintain buffer inventory / redundancy,
- still achieve good supply under contested conditions.

---

### 3.5 Level 5 — Full risk-aware, long-horizon, multi-echelon

**Goal:** Integrate all previous skills and optimize **supply–cost–risk trade-offs** in a more
realistic, long-horizon setting.

- Horizon:

  - Larger (T) (e.g., 20+ periods).

- Network:

  - Multi-echelon structure:

    - base depot(s),
    - several intermediate hubs,
    - multiple frontline demand nodes.

- Demands:

  - Stochastic over time; possibly with regime changes (e.g., “calm” vs “peak” phases).

- Contested:

  - As in Level 4: time-varying availability and risk on edges.

- Objective:

  - Same basic components as Level 4 (supply, cost, loss), but **risk-aware aggregation**:

    - For each scenario:

      - compute fulfilled supply and total loss / cost,

    - across scenarios:

      - use CVaR or other risk-sensitive functional on
        $Z(\pi,\omega) = \gamma C_{\text{trans}} + \beta L_{\text{loss}}$.

  - Example:
    $J_{\text{Lv5}}(\pi) = \mathbb{E}[R_{\text{sup}}(\pi,\omega)] - \lambda \, \text{CVaR}_\tau(Z(\pi,\omega)).$

Level 5 is intended as the **“final boss”** task: the curriculum should help ShinkaEvolve
discover policies that generalize from simple, deterministic settings (Level 1) to this full,
risk-aware, contested logistics environment.

---

## 4. How to Use This Example

- `initial.py` contains:

  - the `LogisticsState` / `Shipments` definitions,
  - a baseline implementation of `logistics_policy(state)`,
  - the simulation loop (`run_experiment`) that runs a full episode for one scenario.

- `evaluate.py`:

  - repeatedly calls `run_experiment` with different seeds / scenarios,
  - aggregates metrics into a scalar `combined_score`,
  - is used by `shinka_launch` or the Python API (`EvolutionRunner`).

Each level can be enabled by:

- changing the **level flag** (e.g. in `get_kwargs` or config),
- which controls:

  - how scenarios are sampled,
  - how `LogisticsState` is populated,
  - which objective (F\_{\text{Lv}k}) is used for scoring.

The unified interface makes it easy to:

- start with **Level 1 only** to debug the pipeline,
- then progressively introduce Levels 2–5 and study the impact of curriculum learning on
  ShinkaEvolve’s behavior and sample efficiency.
