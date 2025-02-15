from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

class NotreDAmeRAG:
    def __init__(self, context_file_path="context.txt"):
        """Initialize the RAG system with the context file path."""
        self.context_file_path = context_file_path
        # Use HuggingFace embeddings instead of OpenAI
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = None
        self.qa_chain = None
        
    def load_and_process_documents(self):
        """Load and process documents into chunks."""
        # Load the document
        loader = TextLoader(self.context_file_path)
        documents = loader.load()
        
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        # Split documents into chunks
        chunks = text_splitter.split_documents(documents)
        return chunks
    
    def create_vector_store(self, chunks):
        """Create and persist the vector store."""
        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        self.vector_store.persist()
        return self.vector_store
    
    def setup_qa_chain(self):
        """Set up the question-answering chain."""
        # Initialize LLM with local server
        llm = OpenAI(
            base_url="http://localhost:1233/v1",  # Local server endpoint
            openai_api_key="dummy_key",  # Placeholder key
            max_tokens=20,
            temperature=0,
        )
        
        # Create retrieval QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 3}  # Retrieve top 3 most relevant chunks
            )
        )
    
    def initialize_system(self):
        """Initialize the complete RAG system."""
        chunks = self.load_and_process_documents()
        self.create_vector_store(chunks)
        self.setup_qa_chain()
    
    def query(self, question: str) -> str:
        """Query the system with a question."""
        if not self.qa_chain:
            raise ValueError("System not initialized. Please run initialize_system() first.")
        
        response = self.qa_chain.run(question)
        return response

def main():
    # Initialize the RAG system
    rag_system = NotreDAmeRAG()
    rag_system.initialize_system()
    
    # Example usage
    while True:
        question = input("\nEnter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
            
        try:
            answer = rag_system.query(question)
            print(f"\nAnswer: {answer}")
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()