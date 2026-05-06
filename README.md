# Multi-Agent Economic Simulation with PPO
 
A from-scratch implementation of a multi-agent reinforcement learning economy. Autonomous agents learn to produce resources, set prices, place bids and asks in a continuous double-auction, and survive a cost-of-living mechanic — entirely through trial and error. No reward signal explicitly teaches them economics. They discover it.
 
The PPO algorithm powering each agent is implemented from scratch in PyTorch. No Stable-Baselines3, no RLlib, no high-level RL libraries. Every component — GAE advantage computation, the clipped surrogate objective, the actor-critic architecture, gradient clipping, on-policy rollouts and updates — is hand-written. The auction matching engine is also custom: a two-pointer market clearing algorithm that handles same-agent skips, partial fills, and atomic balance/inventory updates.
 
## Why this exists
 
Most RL projects benchmark a single agent against a fixed environment. This one asks a different question: what happens when many learning agents inhabit the same environment, each with its own policy, each adapting to the others as they adapt back? The economy isn't given to the agents — it emerges from their interactions. Prices aren't set by a designer — they're discovered by the market. Specialization isn't programmed — it's the equilibrium that learning agents converge to when comparative advantage exists.
 
The simulation has been run across a wide range of initial conditions, and the same kinds of emergent behavior keep showing up. That's the interesting part.
 
## What's emergent about it
 
The configuration is fully parameterized. You can hand the system any number of agents, any number of resources, any pattern of production costs across agents, any starting balances, any initial prices. None of these are baked into the agent's policy. Yet across configurations, recognizable economic behaviors emerge:
 
**Specialization.** When agents have heterogeneous production costs — some agents cheap at one resource, expensive at others — they reliably converge to producing what they're best at. The reward signal never says "produce your cheap resource." It just says "wealth is good." The agents figure out the rest, because producing your expensive resources and trying to compete in the market for them is dominated by producing your cheap resources and trading.
 
**Comparative advantage in action.** Even when one agent is cheaper than another at every resource, both still benefit from trade. The agents learn this. The "less efficient" agent still finds a profitable niche by focusing on its *least disadvantaged* resource. This is Ricardian comparative advantage emerging from gradient descent.
 
**Price discovery.** With no central price-setter, prices stabilize around production costs plus a margin. When supply of a resource is high, its price drops; when demand spikes, prices rise. Volatility decreases as agents' policies stabilize. The market clears.
 
**Hoarding as a transient strategy.** Early in training, some agents stumble into a "produce everything, never sell" policy. It works briefly because net worth includes inventory at market prices. But cost-of-living and the opportunity cost of unsold inventory eventually punish it, and the policy fades.
 
**Trade networks.** With more agents and more resources, you start to see structure in *who trades with whom*. Agents that specialize in complementary resources end up as repeat counterparties. The market has shape, not just throughput.
 
**Adaptation to shocks.** Change the initial conditions mid-training — bump cost of living, alter a resource's starting price, kill an agent — and the surviving agents reorganize. New equilibria form. The policies are robust to perturbations they were never explicitly trained against.
 
## Architecture
 
### The agent
 
Each agent runs an independent PPO policy with a shared-trunk, multi-head architecture:
 
```
observation → trunk (2 hidden layers, ReLU) → ┬→ produce head
                                              ├→ orders head
                                              └→ critic (value head)
```
 
The **produce head** outputs three values per resource: a logit for whether to produce, a Normal mean for quantity, and a Normal log-std for quantity. The agent samples a Bernoulli for the produce/skip decision and a Normal-then-softplus for the quantity itself.
 
The **orders head** outputs six values per resource: act logit (place an order or skip), side logit (bid or ask), price mean and log-std, quantity mean and log-std. Each tick, per resource, the agent samples whether to act, which side, what price, what quantity.
 
The **critic** estimates V(s) for advantage computation.
 
The trunk is shared across heads — features learned for production decisions inform trading decisions and vice versa. Both heads contribute to a single PPO log-prob ratio per timestep.
 
### The observation
 
Each tick, the agent sees:
 
- Its own inventory of every resource
- The last average traded price of every resource (carried forward when no trades)
- The last traded volume of every resource
- Its own balance
- The cost of living
All values are squashed through `tanh(x/scale)` before entering the network. This keeps inputs bounded regardless of how large inventories or balances grow over thousands of ticks. Gradient flow stays smooth, no dead zones, no numerical explosions when an agent stockpiles ten thousand units of something.
 
### The action space
 
Per resource, the agent decides:
 
- Whether to produce, and if so how much
- Whether to place an order, and if so:
  - Bid or ask
  - Price
  - Quantity
For five resources, that's 21 produce-related sub-actions and 30 order-related sub-actions per tick. The full joint action space is enormous, but PPO handles it cleanly because the per-resource sub-actions are conditionally independent given the observation, and their log-probs sum.
 
### PPO from scratch
 
The training loop, written end-to-end:
 
- **Rollout collection**: agents act in the economy for an episode, storing (obs, action, log_prob, value, reward, done) per tick in private buffers
- **Generalized Advantage Estimation**: computed backwards through the trajectory with γ=0.99, λ=0.95, masking at episode boundaries
- **Returns** = advantages + values, used as targets for the critic
- **K epochs of update** per rollout (typically 4):
  - Re-run the network on stored observations to get new log_probs and new values
  - Compute the ratio = exp(new_log_prob - old_log_prob) per timestep
  - Clipped surrogate loss: -min(ratio · A, clip(ratio, 1-ε, 1+ε) · A).mean()
  - Value loss: MSE between new values and stored returns
  - Total loss: policy + 0.5 · value
  - Backprop, gradient norm clipped at 0.5, Adam step at lr=3e-4
Old log-probs and values are detached when stored — they're treated as constants in the PPO objective, which is mathematically required for the ratio to make sense.
 
### The market
 
The Economy class runs the simulation loop and clears the market each tick:
 
1. **Production phase**: every agent independently produces, paying production costs from its balance and updating its own inventory
2. **Order phase**: every agent submits a list of bids and asks
3. **Validation**: orders with negative quantities, negative prices, insufficient balance (for bids), or insufficient inventory (for asks) are rejected before matching
4. **Matching**: bids sorted descending, asks sorted ascending. Two-pointer walk through both lists, matching the highest bid against the lowest ask at the midpoint price. Same-agent pairs are skipped (an agent doesn't trade with itself). Partial fills are handled by decrementing the larger order's quantity and advancing only the smaller one's pointer.
5. **Settlement**: balance and inventory updates happen atomically per match. Average price and traded volume are tracked per resource.
6. **Observation**: every agent sees the tick's market summary, deducts cost-of-living, and computes its reward
7. **Death check**: agents whose balance falls below the minimum threshold are marked dead and skipped in subsequent ticks
The whole simulation is async — agent decisions for a given tick happen concurrently via `asyncio.gather`, which makes it natural to scale to larger agent populations.
 
### Reward design
 
The reward each tick is:
 
```
reward = (net_worth_t - net_worth_{t-1}) - invalid_order_penalty
```
 
where `net_worth = balance + Σ inventory[r] · last_price[r]`. Both gains and losses are captured. The penalty term is the agent's own count of invalid orders or productions it tried to make this tick — it learns to stop attempting actions it can't afford or fulfill.
 
A large negative reward is added on death.
 
This formulation deliberately keeps the reward dense and shaped — sparse reward (e.g., +1 for surviving, 0 otherwise) would be very hard for PPO to learn from in a market this complex. With the dense signal, agents see the consequences of their actions every single tick and credit-assign accordingly.
 
## Configuration
 
Everything is parameterized. You hand `build_agents` and `train_loop` whatever you want:
 
- Number of agents
- Number and names of resources
- Per-agent production cost dictionaries (the heart of the heterogeneity)
- Starting inventories and balances
- Initial market prices
- Cost of living
- Minimum survival balance
- Tick count per episode
- Number of training episodes
Want a five-agent economy with one specialist per resource? Set up the production costs that way. Want a single dominant producer surrounded by traders? Give one agent low costs across the board and the rest high costs. Want scarcity-driven dynamics? Start with low initial inventories and high cost of living. The setup is data, not code.
 
## Repository layout
 
```
economy_sim/
├── main.py            # Order, Trade, Agent base class, Economy with auction
├── model.py           # PPO network (shared trunk + multi-head)
├── example_sim.py     # RealAgent (full PPO logic), build_agents, train_loop
└── README.md
```
 
`main.py` holds the environment-side logic: the order types, the Economy class, the matching engine. It knows nothing about RL.
 
`model.py` holds the neural network architecture. It knows nothing about the economy.
 
`example_sim.py` is where everything connects: the `RealAgent` class that subclasses the abstract `Agent`, implements PPO end-to-end, manages its own rollout buffer, and trains itself. The training loop instantiates agents, runs episodes, and calls `agent.train()` after each one.
 
## Running
 
```bash
pip install torch
python example_sim.py
```
 
Each episode prints final inventories and balances. Watching these evolve across episodes is the most direct way to see learning happen.
 
## Design decisions worth flagging
 
**One PPO update per episode, not per tick.** Per-tick updates would have extreme gradient variance (one transition per update is essentially noise) and would invalidate the on-policy assumption immediately. The buffer accumulates a full episode's worth of trajectory, then trains.
 
**Agents handle their own production accounting.** When an agent produces, it deducts the cost from its balance and adds to its inventory inside `produce()`. The Economy doesn't audit this — it trusts the agent. Trading, where adversarial behavior matters, is fully mediated by the Economy.
 
**Net-worth-based reward instead of balance-only.** Net worth captures the value of inventory at market prices, so an agent stockpiling is rewarded *temporarily* for the inventory's mark-to-market value. But cost-of-living forces continued cash flow, so pure hoarding still loses out. The combination produces interesting dynamics: agents will hold inventory speculatively if prices are rising, sell aggressively if prices are falling.
 
**Squash, don't clip.** Observation values pass through smooth `tanh` rather than hard clamps. The network sees a smooth function of the state, with gradients defined everywhere.
 
**Shared trunk for both action heads and the critic.** Features learned for one decision inform the others. Backprop from all three losses flows into the same trunk parameters.
 
**Detached old log-probs.** Stored log-probs in the buffer are detached at storage time. They're constants in the PPO ratio. Any attempt to backprop through them would be silently wrong, so detaching at the source prevents subtle bugs.
 
## What I learned building this
 
Several non-obvious failure modes that taught me a lot:
 
**Numerical stability isn't optional.** Early versions blew up to NaN within a few episodes because raw observation values (inventories of thousands, balances of tens of thousands) propagated through the network and produced gigantic activations. The fix wasn't smarter math — it was just bounded inputs via squashing. Bounded inputs, gradient clipping, log-std initialization in a sane range. These three together are the difference between training and divergence.
 
**Reward shaping is everything in multi-agent settings.** With pure balance-delta reward, agents learn to do nothing — invalid orders get penalized, valid orders are uncertain, doing nothing is safe. With net-worth reward, they over-produce and hoard. Finding the right balance between immediate balance change, mark-to-market inventory, penalty for invalid actions, and survival incentive is most of the engineering work.
 
**Multi-agent learning is non-stationary by design.** The "environment" changes as the other agents learn. Strategies that work in episode 10 may be exploited in episode 50. Convergence isn't guaranteed and may not even be desired. The fact that the system finds equilibria at all is partly a property of PPO's clipping (which prevents catastrophic drift) and partly luck.
 
**On-policy means on-policy.** Storing actions and log-probs at the moment of the decision, with detach in the right places, with the same network used for sampling and for log-prob recomputation under the new policy — there are a lot of places to introduce subtle off-policy bugs. Getting the bookkeeping right is most of getting PPO right.
 
## What's next
 
The system is a research platform. Some directions it naturally extends:
 
- **Visualization**: per-tick price series, per-agent net worth trajectories, trade flow heatmaps, animation of the market state across an episode
- **Heterogeneous policies**: mix RL agents with rule-based agents to study how PPO learners exploit fixed strategies, or vice versa
- **Communication**: give agents an explicit messaging channel and see whether useful protocols emerge
- **Shocks and regime changes**: change parameters mid-training and study adaptation rates
- **Larger populations**: scale to dozens or hundreds of agents and look for macro-scale phenomena (boom-bust cycles, wealth concentration, market crashes)
- **Resource scarcity over time**: deplete resources or make production costs non-stationary
The infrastructure supports all of this. The PPO is solid, the auction is correct, the agent abstraction is clean. What's left is the science.
 