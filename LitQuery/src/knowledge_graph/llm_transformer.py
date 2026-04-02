from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer


def create_llm_transformer():
    return LLMGraphTransformer(
        llm=ChatOpenAI(temperature=0, model_name="gpt-4o"),
        allowed_nodes=[
            "Concept", "Keyword", "Topic", "Method", "Finding", "Research_Area"
        ],
        allowed_relationships=[
            ("Concept", "RELATED_TO", "Concept"),
            ("Keyword", "BELONGS_TO", "Topic"),
            ("Concept", "PART_OF", "Research_Area"),
            ("Method", "USED_IN", "Research_Area"),
            ("Finding", "SUPPORTS", "Concept"),
            ("Topic", "SUBFIELD_OF", "Research_Area"),
            ("Concept", "STUDIED_WITH", "Method")
        ],
        strict_mode=True,
        node_properties=["importance", "definition"],
        relationship_properties=["strength"]
    )