from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class NodeType(str, Enum):
    """Enumeration of knowledge graph node types."""
    CHALLENGE = "challenge"
    ARTIFACT = "artifact"
    TECHNIQUE = "technique"
    TOOL = "tool"
    CATEGORY = "category"


@dataclass
class Node:
    """Represents a vertex in the knowledge graph."""
    id: str
    type: NodeType
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            **self.attributes
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Node":
        """Create a Node instance from a dictionary."""
        # Extract id and type, everything else is an attribute
        node_id = data.pop("id")
        node_type = NodeType(data.pop("type"))
        return cls(id=node_id, type=node_type, attributes=data)


@dataclass
class Edge:
    """Represents a directed edge between two nodes in the knowledge graph."""
    source: str
    target: str
    weight: float = 0.0
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "source": self.source,
            "target": self.target,
            "weight": self.weight,
            **self.attributes
        }
