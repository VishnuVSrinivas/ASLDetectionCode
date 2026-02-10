#!/usr/bin/env python3

from llm_processor import LLMProcessor

def test_llm_processing():
    """Test LLM processing with sample signs."""
    processor = LLMProcessor()
    
    # Test cases
    test_cases = [
        ['like', 'study', 'language'],
        ['love', 'pizza'],
        ['hate', 'homework'],
        ['book', 'read', 'good']
    ]
    
    for i, signs in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {signs} ---")
        result = processor.process_signs_to_sentence(signs)
        print(f"Sentence: {result['sentence']}")
        print(f"Emoji: {result['emoji']}")
        print(f"Sentiment: {result['sentiment']}")

if __name__ == "__main__":
    test_llm_processing() 