import asyncio
import pytest
import main
from example_sim import RandomAgent, make_economy

def test_conservation():
    result = asyncio.run(make_economy().run())
    total = {}
    for inv in result.inventories.values():
        for r, q in inv.items():
            total[r] = total.get(r, 0) + q
    for r in total:
        assert total[r] == result.production_per_resource.get(r, 0) + sum(
            inv.get(r, 0) for inv in make_economy().inventories.values()
        )

def test_deterministic():
    result1 = asyncio.run(make_economy().run())
    result2 = asyncio.run(make_economy().run())
    assert result1.ledger.log == result2.ledger.log