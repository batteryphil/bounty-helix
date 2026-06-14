"""
This tool periodically reviews beliefs marked as 'stale' and removes any with low confidence scores.
"""

import json
import time
from helix import Tool, ToolRegistry

class BeliefPruner(Tool):
    """
    A tool to periodically review stale beliefs and remove those with low confidence scores.
    """
    
    name = "belief_pruner"
    description = "Removes low-confidence stale beliefs"
    toolset = "self"
    schedule = "0 0 * * *"
    
    def run(self):
        beliefs = self.get_state('beliefs')
        stale_beliefs = [belief for belief in beliefs if belief['stale']]
        
        for belief in stale_beliefs:
            if belief['confidence'] < 0.5:
                beliefs.remove(belief)
                self.set_state({'beliefs': beliefs})
                self.log(f"Removed low-confidence stale belief: {belief}")
        
        self.set_state({'beliefs': beliefs})

def main():
    registry = ToolRegistry()
    registry.register_tool(BeliefPruner)
    registry.run_all()

if __name__ == "__main__":
    main()