import asyncio
import random
import main
import torch
from model import PPO
from torch.distributions import Bernoulli, Normal
from main import Economy
from main import SimulationResult

class RealAgent(main.Agent):

    def __init__(
        self,
        agent_id: str,
        inventory: dict[str, int],
        balance: float,
        resource_names: frozenset[str],
        production_costs: dict[str,float],
        min_requirement: float,
        cost_of_living: float,
        avg_price_per_resource: dict[str, float],
        volume_traded_per_resource: dict[str, int],
        prev_net_worth: float
    ):
        super().__init__(agent_id, inventory, balance, production_costs, cost_of_living)
        self._resource_list = sorted(resource_names)
        self._last_avg_prices: dict[str, float] = avg_price_per_resource
        self.last_volume_traded_per_resource: dict[str, int] = volume_traded_per_resource
        self.model = PPO(len(resource_names))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.0003)
        self.curr_penalty = 0
        self.curr_reward = 0
        self.alive = True
        self.min_requirement = min_requirement
        self.prev_net_worth = prev_net_worth

        self.buffer = {}
        self.buffer["obs"] = []
        self.buffer["produce_actions"] = []
        self.buffer["orders_actions"] = []
        self.buffer["rewards"] = []
        self.buffer["produce_log_prob"] = []
        self.buffer["orders_log_prob"] = []
        self.buffer["values"] = []
        self.buffer["dones"] = []
    
    #Produce a random amount of a random resource for the current tick.
    async def produce(self, tick_idx, seed: int) -> dict[str, int]:

        curr_state = []
        for resource in self._resource_list:
            curr_state.append(self.inventory.get(resource,0))
            curr_state.append(self.last_volume_traded_per_resource.get(resource,0))
            curr_state.append(self._last_avg_prices.get(resource,0))
        curr_state.append(self.balance)
        curr_state.append(self.cost_of_living)

        curr_state = torch.tensor(curr_state, dtype=torch.float32)
    
        self.produce_logits, self.order_logits, self.value = self.model(curr_state)
        
        produce_logits = self.produce_logits.view(self.model.n,3)
        act_logits = produce_logits[:, 0]
        qty_means = produce_logits[:, 1]
        qty_log_stds = produce_logits[:, 2]

        act_dist = Bernoulli(logits=act_logits)
        acts = act_dist.sample()  # (n,) of 0/1

        qty_dist = Normal(qty_means, qty_log_stds.exp())
        qty_raw = qty_dist.sample()  # (n,)
        qtys = torch.nn.functional.softplus(qty_raw)  # ensure >= 0

        produce_total_lp = (
        act_dist.log_prob(acts)
        + qty_dist.log_prob(qty_raw) * acts
        ).sum()

        self.buffer["obs"].append(curr_state)
        self.buffer["produce_log_prob"].append(produce_total_lp.detach())
        self.buffer["produce_actions"].append((acts,qty_raw))
        self.buffer["values"].append(self.value.detach())

        produces = {}
        for i in range(self.model.n):
            if acts[i] == 1:
                if int(qtys[i].item()) * self.production_costs[self._resource_list[i]] > self.balance:
                    self.curr_penalty += int(qtys[i].item()) * self.production_costs[self._resource_list[i]] - self.balance
                    continue
                produces[self._resource_list[i]] = int(qtys[i].item())

        for resource in produces:
            self.inventory[resource] += produces[resource]
            self.balance -= produces[resource] * self.production_costs[resource]
        return produces


    #Submit bids for the most expensive resource and asks for the most traded resource.
    async def submit_orders(
        self, tick_idx, seed, inventory: dict[str, int], balance: float
    ) -> list[main.Order]:
    
        order_logits = self.order_logits.view(self.model.n,6)
        act_logits = order_logits[:, 0]
        bid_logits = order_logits[:, 1]
        price_means = order_logits[:, 2]
        price_log_stds = order_logits[:, 3]
        qty_means = order_logits[:, 4]
        qty_log_stds = order_logits[:, 5]

        act_dist = Bernoulli(logits=act_logits)
        acts = act_dist.sample()  # (n,) of 0/1

        bid_dist = Bernoulli(logits=bid_logits)
        bids = bid_dist.sample()  # (n,) of 0/1

        price_dist = Normal(price_means, price_log_stds.exp())
        price_raw = price_dist.sample()  # (n,)
        prices = torch.nn.functional.softplus(price_raw)  # ensure >= 0

        qty_dist = Normal(qty_means, qty_log_stds.exp())
        qty_raw = qty_dist.sample()  # (n,)
        qtys = torch.nn.functional.softplus(qty_raw)  # ensure >= 0

        orders_total_lp = (
        act_dist.log_prob(acts)
        + bid_dist.log_prob(bids) * acts
        + price_dist.log_prob(price_raw) * acts
        + qty_dist.log_prob(qty_raw) * acts
        ).sum()

        self.buffer["orders_actions"].append((acts,bids,price_raw,qty_raw))
        self.buffer["orders_log_prob"].append(orders_total_lp.detach())

        orders = []
        for i in range(self.model.n):
            if acts[i] == 1:
                if bids[i] == 1:
                    side = "bid"
                else:
                    side = "ask"
                
                if side == "bid":
                    if prices[i].item() > self.balance:
                        self.curr_penalty += prices[i].item() - self.balance
                        continue
                if side == "ask":
                    if qtys[i].item() > self.inventory[self._resource_list[i]]:
                        self.curr_penalty += qtys[i].item() - self.inventory[self._resource_list[i]]
                        continue
                orders.append(main.Order(self.id,self._resource_list[i],side,int(qtys[i].round().item()),prices[i].item()))
        return orders
    
    async def observe(self, tick_idx, trades_list, avg_price_per_resource, volume_traded_per_resource) -> None:
        self._last_avg_prices = dict(avg_price_per_resource)
        self.last_volume_traded_per_resource = dict(volume_traded_per_resource)
        self.balance -= self.cost_of_living

        if self.balance < self.min_requirement:
            self.alive = False
            self.buffer["dones"].append(1)
            self.curr_reward = -100000 
            self.buffer["rewards"].append(self.curr_reward)
            return 

        net_worth = self.balance 
        for resource in self.inventory:
            net_worth += self.inventory[resource] * self._last_avg_prices[resource]

        self.curr_reward = net_worth - self.prev_net_worth
        self.curr_reward -= self.curr_penalty
        self.prev_net_worth = net_worth
        self.curr_penalty = 0
        self.buffer["rewards"].append(self.curr_reward)
        self.buffer["dones"].append(0)

    def compute_gae(self, rewards, values, dones, gamma=0.99, lam=0.95):
        T = len(rewards)
        advantages = torch.zeros(T)
        gae = 0
        for t in reversed(range(T)):
            next_value = 0 if t == T - 1 else values[t + 1]
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * mask - values[t]
            gae = delta + gamma * lam * mask * gae
            advantages[t] = gae
        returns = advantages + values
        return returns, advantages
    
    def recompute(self, obs):
        produce_logits, order_logits, values = self.model(obs)
        
        # produce
        pl = produce_logits.view(-1, self.model.n, 3)
        p_acts, p_qty_raw = zip(*self.buffer["produce_actions"])
        p_acts = torch.stack(p_acts)
        p_qty_raw = torch.stack(p_qty_raw)
        p_act_dist = Bernoulli(logits=pl[:, :, 0])
        p_qty_dist = Normal(pl[:, :, 1], pl[:, :, 2].exp())
        new_produce_lp = (
            p_act_dist.log_prob(p_acts)
            + p_qty_dist.log_prob(p_qty_raw) * p_acts
        ).sum(dim=1)
        
        # orders
        ol = order_logits.view(-1, self.model.n, 6)
        o_acts, o_bids, o_price_raw, o_qty_raw = zip(*self.buffer["orders_actions"])
        o_acts = torch.stack(o_acts)
        o_bids = torch.stack(o_bids)
        o_price_raw = torch.stack(o_price_raw)
        o_qty_raw = torch.stack(o_qty_raw)
        o_act_dist = Bernoulli(logits=ol[:, :, 0])
        o_bid_dist = Bernoulli(logits=ol[:, :, 1])
        o_price_dist = Normal(ol[:, :, 2], ol[:, :, 3].exp())
        o_qty_dist = Normal(ol[:, :, 4], ol[:, :, 5].exp())
        new_orders_lp = (
            o_act_dist.log_prob(o_acts)
            + o_bid_dist.log_prob(o_bids) * o_acts
            + o_price_dist.log_prob(o_price_raw) * o_acts
            + o_qty_dist.log_prob(o_qty_raw) * o_acts
        ).sum(dim=1)
        
        return new_produce_lp, new_orders_lp, values.squeeze(-1)

    async def train(self) -> None:
        obs = torch.stack(self.buffer["obs"])  # (T, obs_dim)
        rewards = torch.tensor(self.buffer["rewards"], dtype=torch.float32)
        values = torch.stack(self.buffer["values"]).squeeze(-1)  # (T,)
        dones = torch.tensor(self.buffer["dones"], dtype=torch.float32)
        old_produce_lp = torch.stack(self.buffer["produce_log_prob"])
        old_orders_lp = torch.stack(self.buffer["orders_log_prob"])

        returns, advantages = self.compute_gae(rewards, values, dones)

        K_epochs = 4
        CLIP = 0.2
        for _ in range(K_epochs):
            new_produce_lp, new_orders_lp, new_values = self.recompute(obs)
            new_lp = new_produce_lp + new_orders_lp
            old_lp = old_produce_lp + old_orders_lp

            ratio = (new_lp - old_lp).exp()
            surr1 = ratio * advantages
            surr2 = ratio.clamp(1 - CLIP, 1 + CLIP) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = (new_values - returns).pow(2).mean()

            loss = policy_loss + 0.5 * value_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        for k in self.buffer:
            self.buffer[k].clear()
    
    def reset(self, balance, inventory, avg_price_per_resource, volume_traded_per_resource, prev_net_worth):
        self.alive = True
        self.curr_reward = 0
        self.curr_penalty = 0
        self.prev_net_worth = prev_net_worth

        self.balance = balance
        self.inventory = inventory
        self._last_avg_prices = avg_price_per_resource
        self.last_volume_traded_per_resource = volume_traded_per_resource
    

def build_agents(n_agents, n_resources, resource_types):
    names = frozenset(resource_types)

    resources_list = sorted(resource_types)  # deterministic order
    production_costs_per_agent = {}
    starting_inventory = {r: 10 for r in resources_list}
    balances = {}
    inventories = {}
    cost_of_living = 10.0
    min_requirement = 100.0
    net_worth = 1000.0 + 10*50.0*n_agents
    for i in range(n_agents):
        agent_id = chr(ord("A") + i)
        cheap_resource = resources_list[i % len(resources_list)]
        costs = {}
        for r in resources_list:
            if r == cheap_resource:
                costs[r] = 1.0   # cheap
            else:
                costs[r] = 5.0   # average
        production_costs_per_agent[agent_id] = costs
        inventories[agent_id] = dict(starting_inventory)
        balances[agent_id] = 1000.0
    initial_prices = {r: 50.0 for r in resources_list}
    initial_volumes = {r: 0 for r in resources_list}

    agents = []
    for i in range(n_agents):
        agent_id = chr(ord("A") + i)
        balance = 1000.0
        agents.append(RealAgent(
            agent_id,
            inventories[agent_id],
            balance,
            names,
            production_costs_per_agent[agent_id],
            min_requirement,
            cost_of_living,
            initial_prices,
            initial_volumes,
            prev_net_worth=net_worth
        ))
    return inventories, balances, agents

def reset_agents(agents, balances, inventories, avg_price_per_resource, volume_traded_per_resource, prev_net_worth):
    for agent in agents:
        agent.reset(balances[agent.id], inventories[agent.id], avg_price_per_resource, volume_traded_per_resource, prev_net_worth)


async def train_loop():
    n_epochs = 100
    n_agents = 5
    n_resources = 5
    resource_types = {"gold", "silver", "wood", "coal", "oil"}
    resources_list = sorted(list(resource_types))  # deterministic order
    inventories, balances, agents = build_agents(n_agents, n_resources, resource_types)
    initial_prices = {r: 50.0 for r in resources_list}
    initial_volumes = {r: 0 for r in resources_list}
    net_worth = 1000.0 + 10*50.0*n_agents
    for _ in range(n_epochs):
        reset_agents(agents, balances, inventories, initial_prices, initial_volumes, net_worth)
        economy = Economy(agents, resource_types, tick_count=100, seed=42, cost_of_living=10.0, avg_price_per_resource=initial_prices, volume_traded_per_resource=initial_volumes)
        result = await economy.run()
        print(result.inventories)
        print(result.balances)
        await asyncio.gather(*[agent.train() for agent in agents])

if __name__ == "__main__":
    asyncio.run(train_loop())
    
