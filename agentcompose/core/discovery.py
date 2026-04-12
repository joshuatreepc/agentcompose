"""
Agent discovery system.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple


class AgentDiscovery:
    """Discovers available agents on disk."""

    def __init__(self, agents_dir: Path | None = None):
        self.agents_dir = agents_dir or Path(__file__).parent

    def discover_agents(self) -> Dict[str, Path]:
        """
        Discover available agents.

        Walks `<agents_dir>/<domain>/<agent>/` and returns each agent directory
        keyed by agent name.

        Returns:
            Dict[agent_name, agent_path]
        """
        agents: Dict[str, Path] = {}

        for domain_dir in self.agents_dir.iterdir():
            if not domain_dir.is_dir() or domain_dir.name.startswith((".", "__")):
                continue

            for agent_dir in domain_dir.iterdir():
                if agent_dir.is_dir() and not agent_dir.name.startswith((".", "__")):
                    agents[agent_dir.name] = agent_dir

        return agents

    def get_agent_info(self, agent: str) -> Optional[Dict]:
        """Get info for a specific agent."""
        agents = self.discover_agents()

        if agent in agents:
            return {"name": agent, "path": str(agents[agent])}

        return None

    def resolve_agent(self, agent_ref: str) -> Tuple[Optional[str], Optional[Path]]:
        """
        Resolve an agent reference to (name, path).

        Args:
            agent_ref: agent name

        Returns:
            Tuple of (agent_name, agent_path) or (None, None)
        """
        agents = self.discover_agents()

        if agent_ref in agents:
            return agent_ref, agents[agent_ref]

        return None, None

    def list_agents(self) -> List[str]:
        """List all available agents."""
        return sorted(self.discover_agents().keys())
