import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import time

class Order:
    def __init__(self,agent_id,resource,side,quantity,price):
        self.agent_id = agent_id
        self.resource = resource
        self.side = side
        self.quantity = quantity
        self.price = price
    

@dataclass(frozen=True)
class Produce:
    tick_idx:int
    agent_id: str
    resource: str
    quantity: int


@dataclass(frozen=True)
class Trade:
    tick_idx: int
    buyer_id: str
    seller_id: str
    resource: str
    quantity: int
    price: float

class Agent(ABC):
    def __init__(self, id: str, inventory: dict[str,int], balance: float, production_costs: dict[str,float], cost_of_living: float):
        self.id = id
        self.inventory = inventory
        self.balance = balance
        self.production_costs = production_costs
        self.cost_of_living = cost_of_living
    
    @abstractmethod
    async def produce(self, tick_idx,seed: int) -> dict[str,int]:
        pass
    
    @abstractmethod
    async def submit_orders(self,tick_idx,seed,inventory,balance) -> list[Order]:
        pass
    
    @abstractmethod
    async def observe(self,tick_idx, trades_list, avg_price_per_resource, volume_traded_per_resource) -> None:
        pass
    
    @abstractmethod
    async def train(self) -> None:
        pass

class SimulationResult:
    def __init__(self,inventories: dict[str,dict[str:int]], balances: dict[str,float], production_per_resource: dict[str,int], trade_volume_per_resource: dict[str,int], ticks: int):
        self.ticks = ticks
        self.inventories = inventories
        self.balances = balances
        self.trade_volume_per_resource = trade_volume_per_resource
        self.production_per_resource = production_per_resource

class Economy:
    def __init__(self,agents: list[Agent],resource_types: set[str], tick_count: int, seed: int, cost_of_living: float, avg_price_per_resource: dict[str:float], volume_traded_per_resource: dict[str:int], timeout_seconds: Optional[float] = None):
        self.agents = agents
        self.all_agents = agents
        self.resource_types = resource_types
        self.tick_count = tick_count
        self.seed = seed 
        self.timeout_seconds = timeout_seconds
        self.cost_of_living = cost_of_living

        self.avg_price_per_resource = avg_price_per_resource
        self.volume_traded_per_resource = volume_traded_per_resource

        #Agent map maps the agent id to the agent object.
        self._agent_map = {}
        for agent in self.agents:
            self._agent_map[agent.id] = agent
                
    async def run(self):
        start = time.time()
        no_ticks = 0
        for tick_idx in range(self.tick_count):

            trades_list = []
            self.agents = [agent for agent in self.all_agents if agent.alive]
            if len(self.agents) == 0:
                break

            avg_price_per_resource = {}
            volume_traded_per_resource = {}
            produced = await asyncio.gather(*[self.agents[i].produce(tick_idx,self.seed+tick_idx*1000 + i) for i in range(len(self.agents))])
                        
            #resource_order_map maps each resource to a dictionary with two keys, 'asks' and 'bids'.
            resource_order_map = {}
            orders = await asyncio.gather(*[self.agents[i].submit_orders(tick_idx,self.seed+tick_idx*1000+i,self.agents[i].inventory,self.agents[i].balance) for i in range(len(self.agents))])
            orders = [item for sublist in orders for item in sublist]
            for order in orders:
                #Reject orders with negative quantity or price.
                if order.quantity <= 0 or order.price <= 0:
                    continue

                agent = self._agent_map[order.agent_id]    
                #Reject orders with insufficient inventory or insufficient balance.
                if order.side == "ask":
                    if order.resource not in agent.inventory or order.quantity > agent.inventory[order.resource]:
                        continue
                else:
                    if agent.balance < order.quantity * order.price:
                        continue

                #Add the order to the resource order map.
                if order.resource not in resource_order_map:
                    resource_order_map[order.resource] = {}
                    resource_order_map[order.resource]['asks'] = []
                    resource_order_map[order.resource]['bids'] = []
                if order.side == "ask":
                    resource_order_map[order.resource]['asks'].append(order)
                else:
                   resource_order_map[order.resource]['bids'].append(order) 
            
            #Market clearing algorithm.
            for resource in resource_order_map:
                #Retrieve the asks and bids for the current resource.
                asks = resource_order_map[resource]["asks"]
                bids = resource_order_map[resource]["bids"]
                #Sort the asks and bids by price.
                asks.sort(key=lambda x: x.price)
                bids.sort(key=lambda x: x.price, reverse = True)

                #Two pointers are used to match highest bids against the lowest asks
                pointer1 = 0
                pointer2 = 0
                while pointer1 < len(bids) and pointer2 < len(asks):
                    tmp_ptr = None
                    curr_bid = bids[pointer1]
                    curr_ask = asks[pointer2]

                    if curr_ask == None:
                        pointer2 += 1
                        continue

                    if curr_bid.price < curr_ask.price:
                        break

                    #Same agents situation, advance through asks until you find a valid pair, while keeping track of pointer2
                    if curr_bid.agent_id == curr_ask.agent_id:
                        tmp_ptr = pointer2 + 1
                        while tmp_ptr < len(asks) and asks[tmp_ptr] is not None and curr_bid.agent_id == asks[tmp_ptr].agent_id:
                            tmp_ptr += 1
        
                        if tmp_ptr == len(asks) or asks[tmp_ptr] is None or asks[tmp_ptr].price > curr_bid.price:
                            pointer1 += 1
                            continue
                        curr_ask = asks[tmp_ptr]

                    ask_agent = self._agent_map[curr_ask.agent_id]
                    bid_agent = self._agent_map[curr_bid.agent_id]
                    trade_price = (curr_ask.price + curr_bid.price) / 2
                    
                    #Reject trades with insufficient inventory.
                    if ask_agent.inventory[resource] < min(curr_ask.quantity,curr_bid.quantity):
                        if tmp_ptr == None:
                            pointer2 += 1
                        continue
                    #Reject trades with insufficient balance.
                    if bid_agent.balance < trade_price * min(curr_ask.quantity,curr_bid.quantity):
                        pointer1 += 1
                        continue

                    #Update the inventories and balances of the ask and bid agents.
                    ask_agent.balance += trade_price * min(curr_ask.quantity,curr_bid.quantity)
                    ask_agent.inventory[resource] -= min(curr_ask.quantity,curr_bid.quantity)
                    bid_agent.balance -= trade_price * min(curr_ask.quantity,curr_bid.quantity)
                    bid_agent.inventory[resource] += min(curr_ask.quantity,curr_bid.quantity)

                    #Update market summary statistics like volume traded and average price per resource.
                    if resource not in volume_traded_per_resource:
                        volume_traded_per_resource[resource] = 0
                    volume_traded_per_resource[resource] += min(curr_bid.quantity,curr_ask.quantity)
                    if resource not in avg_price_per_resource:
                        avg_price_per_resource[resource] = [0,0]
                    avg_price_per_resource[resource][0] += 1
                    avg_price_per_resource[resource][1] += trade_price
                    
                    if curr_bid.quantity > curr_ask.quantity:
                        curr_bid.quantity -= curr_ask.quantity
                        if tmp_ptr == None:
                            pointer2 += 1
                    elif curr_bid.quantity < curr_ask.quantity:
                        curr_ask.quantity -= curr_bid.quantity
                        pointer1 += 1
                    else:
                        pointer1 += 1
                        if tmp_ptr == None:
                            pointer2 += 1
                    
                    if tmp_ptr:
                        asks[tmp_ptr] = None
                    
            
            #Reduce every agents balance by their cost of living

            avg_price_per_resource = {r: v[1]/v[0] for r, v in avg_price_per_resource.items()}
            for resource in avg_price_per_resource:
                self.avg_price_per_resource[resource] = avg_price_per_resource[resource]
            for resource in volume_traded_per_resource:
                self.volume_traded_per_resource[resource] = volume_traded_per_resource[resource]
            await asyncio.gather(*[self.agents[i].observe(tick_idx,trades_list,self.avg_price_per_resource,self.volume_traded_per_resource) for i in range(len(self.agents))])
            
            #Assert the conservation invariant for each resource.
            no_ticks += 1
            if self.timeout_seconds and time.time()-start > self.timeout_seconds:
                break 

        #Create a map of the final inventories and balances of each agent.
        balances_map = {}
        inventory_map = {}
        for agent in self._agent_map:
            inventory_map[agent] = self._agent_map[agent].inventory
            balances_map[agent] = self._agent_map[agent].balance
        
        #Return SimulationResult object 
        sim_result = SimulationResult(inventory_map, balances_map, self.avg_price_per_resource, self.volume_traded_per_resource, no_ticks)

        return sim_result


    
    
