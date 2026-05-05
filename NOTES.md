Assumptions:

In the case where two agents are matched for a bid and ask, and they happen to be the same agent, it is unclear whether we advance to the next bid to find a different agent for the current ask, or do we advance to the next ask to find a different agent for the same bid. I went with the second choice and chose to prioritise the bid and find a next ask for the bid that matches it.

In addition, when a trade fails due to lack of resources in the ask agent's inventory or lack of money in the bid agent's balance, perhaps if we were to check the next bidder for the current ask agent or the next ask agent for the current bidder, they might have been able to have a succesfful trade. However, that required the use of another temporary pointer so that we can go back to the original pointer's spot to proceed. And it just so happens that that messed up the code structure, no matter how I tried to do it. So unfortunately, when the bidder does not have enough money for a trade, I skipped them completely. And when an ask agent does not have enough inventory for a sell, they were skipped completely. Looking back, maybe I could have looped through the bid and ask arrays over and over or recursively did the same process on them until no bid matched with any ask and vice verse, but unfortunately I did not have enough time and this was an unlikely scenario that wasn't worth the limited time.

Testing:

xample_sim.py generates an example economy of 10 random agents. test_economy.py runs tests on that example with pytest test_economy.py 


