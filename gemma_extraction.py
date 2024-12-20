import streamlit as st
from langchain_openai import OpenAI
from datasets import load_dataset
import time

def load_local_llm():
    """Initialize the Gemma LLM"""
    return OpenAI(
        base_url="http://localhost:1233/v1",
        openai_api_key="dummy_key",
        max_tokens=200,
        temperature=0.1
    )

def load_squad_data(num_samples=100):
    """Load and prepare SQuAD dataset"""
    print("Loading SQuAD dataset...")
    dataset = load_dataset("rajpurkar/squad")
    samples = dataset["train"].select(range(num_samples))
    
    # Extract unique contexts
    contexts = list(set([entry["context"] for entry in samples]))
    print(f"Loaded {len(contexts)} unique contexts")
    return contexts

def extract_triples(llm, text):
    """Extract triples from text using Gemma model"""
    prompt = """Extract factual (head entity, relation, tail entity) triples from this text.
    Format each triple as: head_entity | relation | tail_entity
    Only include relationships explicitly stated in the text.
    
    Text: {text}
    
    Triples:"""
    
    try:
        response = llm.invoke(prompt.format(text=text))
        # Filter valid triples
        triples = [
            triple.strip() 
            for triple in response.strip().split('\n') 
            if triple.strip() and '|' in triple
        ]
        return triples
    except Exception as e:
        print(f"Error during extraction: {e}")
        return []

def process_text(text, llm):
    """Process text by splitting into sentences and extracting triples"""
    # Simple sentence splitting
    sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
    
    print("\nProcessing text:")
    print("-" * 80)
    print(text)
    print("-" * 80)
    
    all_triples = []
    
    for i, sentence in enumerate(sentences, 1):
        print(f"\nSentence {i}: {sentence}")
        triples = extract_triples(llm, sentence)
        
        if triples:
            print("Extracted triples:")
            for triple in triples:
                print(f"  → {triple}")
                all_triples.append({
                    'sentence': sentence,
                    'triple': triple
                })
        else:
            print("No triples found in this sentence.")
            
    return all_triples

def main():
    print("Initializing triple extraction system...")
    
    # Initialize LLM
    llm = load_local_llm()
    print("LLM initialized successfully")
    
    # Load dataset
    contexts = load_squad_data(num_samples=10)  # Start with 10 samples for testing
    
    # Process each context
    all_results = []
    for i, context in enumerate(contexts, 1):
        print(f"\n{'='*80}")
        print(f"Processing Context {i}/{len(contexts)}")
        print(f"{'='*80}")
        
        triples = process_text(context, llm)
        all_results.extend(triples)
        
        # Print summary for this context
        print(f"\nSummary for Context {i}:")
        print(f"Found {len(triples)} triples")
        
        # Optional: Add a small delay to make output more readable
        time.sleep(1)
    
    # Print final summary
    print("\n" + "="*80)
    print("EXTRACTION COMPLETE")
    print(f"Total contexts processed: {len(contexts)}")
    print(f"Total triples extracted: {len(all_results)}")
    print("="*80)
    
    # Print all unique triples at the end
    unique_triples = set(item['triple'] for item in all_results)
    print("\nAll unique triples extracted:")
    for triple in sorted(unique_triples):
        print(f"→ {triple}")

if __name__ == "__main__":
    main()