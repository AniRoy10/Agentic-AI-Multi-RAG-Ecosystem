import os
from langchain_groq import ChatGroq
from hr_pipeline import hr_pipeline
from claim_pipeline import claim_pipeline
from reimbursement_pipeline import reimbursement_pipeline

class MetaAgent:
    def __init__(self):
        """Initialize the meta-agent with LLM and RAG pipelines."""
        self.classifier_llm = ChatGroq(model_name="llama-3.3-70b-versatile")
        self.rag_pipelines = {
            "HR Policy": hr_pipeline,
            "Claims Policy": claim_pipeline,
            "Reimbursement Policy": reimbursement_pipeline,
        }

    def classify_query(self, query):
        """Classify the query into a document type."""
        prompt = f"""
        You are a routing assistant. Based on the query below, classify it into one of the following categories:
        1. HR Policy
        2. Claims Policy
        3. Reimbursement Policy

        Query: "{query}"

        Answer with just the category name (e.g., "HR Policy").
        """
        response = self.classifier_llm.invoke(prompt)
        
        if hasattr(response, 'content'):
            category = response.content.strip()
        else:
            category = str(response).strip()
        
        valid_categories = {"HR Policy", "Claims Policy", "Reimbursement Policy"}
        if category not in valid_categories:
            print(f"Unexpected category response: {category}")
            return None

        return category

    def route_query(self, query):
        """Route the query to the appropriate RAG pipeline."""
        category = self.classify_query(query)
        if category in self.rag_pipelines:
            print(f"Routing query to the {category} RAG pipeline.")
            rag_pipeline = self.rag_pipelines[category]
            return rag_pipeline(query)
        else:
            print(f"Unrecognized category: {category}")
            return "Sorry, I couldn't classify your query. Please try rephrasing."


# Example Usage
if __name__ == "__main__":
    # Create an instance of the MetaAgent
    meta_agent = MetaAgent()

    # Example query
    user_query = "How many annual leaves are allowed?"
    response = meta_agent.route_query(user_query)
    print("\n=== Answer ===\n")
    print(response.content)
    print("\n==============\n")
