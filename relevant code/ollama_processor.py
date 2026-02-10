#!/usr/bin/env python3

import json
import requests
import os
from typing import List, Dict, Optional

class OllamaProcessor:
    def __init__(self, model: str = "llama2:7b", base_url: str = "http://localhost:11434"):
        """Initialize Ollama processor."""
        self.model = model
        self.base_url = base_url
        self.allowed_emojis = "😊 🤩 😢 😠 😕 😬 😐 😲 ❤️ 😴"
        
        # Test connection
        try:
            response = requests.get(f"{base_url}/api/tags")
            if response.status_code != 200:
                print(f"Warning: Could not connect to Ollama at {base_url}")
        except Exception as e:
            print(f"Warning: Ollama not running at {base_url}: {e}")
    
    def process_signs_to_sentence(self, signs: List[str]) -> Dict:
        """
        Convert a list of predicted signs into a cleaned sentence with emoji and sentiment.
        
        Args:
            signs: List of predicted sign labels (e.g., ['play', 'you', 'want'])
            
        Returns:
            Dictionary with sentence, emoji, and sentiment
        """
        # Join signs into rough input
        signs_text = " ".join(signs)
        
        prompt = f"""You are a helpful assistant that converts ASL signs into proper English sentences and adds appropriate emojis and sentiment analysis.

Input signs: {signs_text}

Instructions:
1. Convert the input signs into a grammatically correct English sentence.
2. Based only on the emotional tone of the sentence, pick one emoji from this list: {self.allowed_emojis}
3. Classify the sentiment of the sentence as: positive, neutral, or negative.
4. Respond in this exact format:

Sentence: <cleaned sentence>
Emoji: <emoji from the list above>
Sentiment: <positive / neutral / negative>

Example:
Input signs: love pizza
Response:
Sentence: I love pizza
Emoji: 😊
Sentiment: positive

Now process the input signs: {signs_text}"""
        
        try:
            response = self._call_ollama(prompt)
            return self._parse_llm_response(response, signs)
            
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return self._fallback_processing(signs)
    
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API."""
        url = f"{self.base_url}/api/generate"
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        
        response = requests.post(url, json=data)
        if response.status_code == 200:
            return response.json()["response"]
        else:
            raise Exception(f"Ollama API error: {response.status_code}")
    
    def _parse_llm_response(self, response_text: str, original_signs: List[str]) -> Dict:
        """Parse the LLM response into structured format."""
        try:
            lines = response_text.strip().split('\n')
            result = {
                'signs': original_signs,
                'sentence': '',
                'emoji': '',
                'sentiment': ''
            }
            
            for line in lines:
                line = line.strip()
                if line.startswith('Sentence:'):
                    result['sentence'] = line.replace('Sentence:', '').strip()
                elif line.startswith('Emoji:'):
                    result['emoji'] = line.replace('Emoji:', '').strip()
                elif line.startswith('Sentiment:'):
                    result['sentiment'] = line.replace('Sentiment:', '').strip()
            
            # Validate emoji is from allowed list
            if result['emoji'] and result['emoji'] not in self.allowed_emojis:
                result['emoji'] = '😐'  # Default to neutral
            
            # Validate sentiment
            if result['sentiment'] not in ['positive', 'neutral', 'negative']:
                result['sentiment'] = 'neutral'
            
            return result
            
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return self._fallback_processing(original_signs)
    
    def _fallback_processing(self, signs: List[str]) -> Dict:
        """Fallback processing when Ollama is not available."""
        # Simple fallback: just join signs and use neutral emoji
        sentence = " ".join(signs).capitalize()
        
        # Basic sentiment analysis based on common words
        positive_words = ['love', 'like', 'good', 'happy', 'great', 'wonderful', 'amazing', 'beautiful']
        negative_words = ['hate', 'bad', 'terrible', 'awful', 'sad', 'angry', 'dislike']
        
        sentiment = 'neutral'
        emoji = '😐'
        
        signs_lower = [s.lower() for s in signs]
        for sign in signs_lower:
            if sign in positive_words:
                sentiment = 'positive'
                emoji = '😊'
                break
            elif sign in negative_words:
                sentiment = 'negative'
                emoji = '😢'
                break
        
        return {
            'signs': signs,
            'sentence': sentence,
            'emoji': emoji,
            'sentiment': sentiment
        }
    
    def reprocess_sentence(self, custom_sentence: str) -> Dict:
        """Reprocess a custom sentence with emoji and sentiment."""
        prompt = f"""Analyze this sentence and provide an appropriate emoji and sentiment:

Sentence: {custom_sentence}

Instructions:
1. Pick one emoji from this list: {self.allowed_emojis}
2. Classify the sentiment as: positive, neutral, or negative.
3. Respond in this format:

Emoji: <emoji from the list above>
Sentiment: <positive / neutral / negative>"""
        
        try:
            response = self._call_ollama(prompt)
            return self._parse_reprocess_response(response, custom_sentence)
        except Exception as e:
            print(f"Error reprocessing sentence: {e}")
            return {
                'sentence': custom_sentence,
                'emoji': '😐',
                'sentiment': 'neutral'
            }
    
    def _parse_reprocess_response(self, response_text: str, original_sentence: str) -> Dict:
        """Parse the reprocess response."""
        try:
            lines = response_text.strip().split('\n')
            result = {
                'sentence': original_sentence,
                'emoji': '😐',
                'sentiment': 'neutral'
            }
            
            for line in lines:
                line = line.strip()
                if line.startswith('Emoji:'):
                    emoji = line.replace('Emoji:', '').strip()
                    if emoji in self.allowed_emojis:
                        result['emoji'] = emoji
                elif line.startswith('Sentiment:'):
                    sentiment = line.replace('Sentiment:', '').strip()
                    if sentiment in ['positive', 'neutral', 'negative']:
                        result['sentiment'] = sentiment
            
            return result
            
        except Exception as e:
            print(f"Error parsing reprocess response: {e}")
            return {
                'sentence': original_sentence,
                'emoji': '😐',
                'sentiment': 'neutral'
            }

def process_inference_results(inference_file: str, output_file: str = None) -> Dict:
    """Process inference results with Ollama LLM."""
    try:
        # Load inference results
        if isinstance(inference_file, str):
            with open(inference_file, 'r') as f:
                inference_results = json.load(f)
        else:
            inference_results = inference_file
        
        # Extract signs from segments
        signs = []
        for segment in inference_results['segments']:
            if segment['predictions']:
                signs.append(segment['predictions'][0]['label'])
        
        # Process with Ollama
        processor = OllamaProcessor()
        result = processor.process_signs_to_sentence(signs)
        
        # Save results if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
        
        return result
        
    except Exception as e:
        print(f"Error processing inference results: {e}")
        return {
            'signs': [],
            'sentence': '',
            'emoji': '😐',
            'sentiment': 'neutral'
        }

if __name__ == "__main__":
    # Test the processor
    processor = OllamaProcessor()
    
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