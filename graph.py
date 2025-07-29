import logging
from models import ResearchState
from nodes import Nodes
from config import Config
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

logger = logging.getLogger(__name__)

class CatalogResearchGraph:
    def __init__(self):
        self.llm = Config.get_llm()
        self.nodes = Nodes(self.llm)
        self.memory = InMemorySaver()
    
    def build(self):
        graph_builder = StateGraph(ResearchState)
        
        graph_builder.add_node("research", self.nodes.research_node)
        graph_builder.add_node("verification", self.nodes.verification_node)
        graph_builder.add_node("followup_research", self.nodes.followup_research_node)
        graph_builder.add_node("output_generation", self.nodes.output_generation_node)
        
        graph_builder.add_edge(START, "research")
        graph_builder.add_edge("research", "verification")
        graph_builder.add_edge("followup_research", "verification")
        
        graph_builder.add_conditional_edges(
            "verification",
            self.nodes.decision_node,
            {"output_generation": "output_generation", "followup_research": "followup_research"}
        )
        
        graph_builder.add_edge("output_generation", END)
        
        graph = graph_builder.compile(checkpointer=self.memory)
        logger.info("Catalog Research graph compiled successfully")
        return graph 
    