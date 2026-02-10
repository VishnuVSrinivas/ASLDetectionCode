#!/usr/bin/env python3

import json
import openai
import os
from typing import List, Dict, Optional

class LLMProcessor:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize LLM processor with OpenAI API key."""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if self.api_key:
            openai.api_key = self.api_key
        else:
            print("Warning: No OpenAI API key provided. Set OPENAI_API_KEY environment variable.")
        
        self.allowed_emojis = "😊 🤩 😢 😠 😕 😬 😐 😲 ❤️ 😴"
    
    def process_signs_to_sentence(self, signs: List[str]) -> Dict:
        """
        Convert a list of predicted signs into a cleaned sentence with emoji and sentiment.
        
        Args:
            signs: List of predicted sign labels (e.g., ['play', 'you', 'want'])
            
        Returns:
            Dictionary with sentence, emoji, and sentiment
        """
        if not self.api_key:
            return self._fallback_processing(signs)
        
        # Join signs into rough input
        signs_text = " ".join(signs)
        
        prompt = f"""
Input signs: {signs_text}

Instructions:
1. Convert the input signs into a grammatically correct English sentence.
2. Based only on the emotional tone of the sentence, pick one emoji from this list:
{self.allowed_emojis}
(Do NOT use any other emoji.)
3. Classify the sentiment of the sentence as: positive, neutral, or negative.
4. Respond in this format:

Sentence: <cleaned sentence>
Emoji: <emoji from the list above>
Sentiment: <positive / neutral / negative>
"""
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            
            result_text = response['choices'][0]['message']['content']
            return self._parse_llm_response(result_text, signs)
            
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return self._fallback_processing(signs)
    
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
            
            return result
            
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return self._fallback_processing(original_signs)
    
    def _fallback_processing(self, signs: List[str]) -> Dict:
        """Fallback processing when API is not available."""
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
            'emoji': '😐',
            'sentiment': 'neutral'
        }
    
    def reprocess_sentence(self, custom_sentence: str) -> Dict:
        """
        Reprocess a custom sentence with emoji and sentiment analysis.
        
        Args:
            custom_sentence: User-edited sentence
            
        Returns:
            Dictionary with sentence, emoji, and sentiment
        """
        if not self.api_key:
            return {
                'sentence': custom_sentence,
                'emoji': '😐',
                'sentiment': 'neutral'
            }
        
        prompt = f"""
Input sentence: {custom_sentence}

Instructions:
1. Based only on the emotional tone of the sentence, pick one emoji from this list:
{self.allowed_emojis}
(Do NOT use any other emoji.)
2. Classify the sentiment of the sentence as: positive, neutral, or negative.
3. Respond in this format:

Emoji: <emoji from the list above>
Sentiment: <positive / neutral / negative>
"""
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            
            result_text = response['choices'][0]['message']['content']
            return self._parse_reprocess_response(result_text, custom_sentence)
            
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return {
                'sentence': custom_sentence,
                'emoji': '😐',
                'sentiment': 'neutral'
            }
    
    def _parse_reprocess_response(self, response_text: str, original_sentence: str) -> Dict:
        """Parse the reprocess LLM response."""
        try:
            lines = response_text.strip().split('\n')
            result = {
                'sentence': original_sentence,
                'emoji': '',
                'sentiment': ''
            }
            
            for line in lines:
                line = line.strip()
                if line.startswith('Emoji:'):
                    result['emoji'] = line.replace('Emoji:', '').strip()
                elif line.startswith('Sentiment:'):
                    result['sentiment'] = line.replace('Sentiment:', '').strip()
            
            # Validate emoji is from allowed list
            if result['emoji'] and result['emoji'] not in self.allowed_emojis:
                result['emoji'] = '😐'
            
            return result
            
        except Exception as e:
            print(f"Error parsing reprocess response: {e}")
            return {
                'sentence': original_sentence,
                'emoji': '😐',
                'sentiment': 'neutral'
            }

def process_inference_results(inference_file: str, output_file: str = None) -> Dict:
    """
    Process inference results and generate sentence with LLM.
    
    Args:
        inference_file: Path to enhanced inference results JSON
        output_file: Optional output file path
        
    Returns:
        Dictionary with processed results
    """
    try:
        with open(inference_file, 'r') as f:
            inference_data = json.load(f)
        
        # Extract top predictions for each segment
        signs = []
        for segment in inference_data['segments']:
            if segment['predictions']:
                # Take the top prediction
                top_pred = segment['predictions'][0]['label']
                signs.append(top_pred)
        
        # Process with LLM
        llm_processor = LLMProcessor()
        result = llm_processor.process_signs_to_sentence(signs)
        
        # Add segment information
        result['segments'] = inference_data['segments']
        result['session_dir'] = inference_data['session_dir']
        
        # Save results
        if output_file is None:
            output_file = inference_file.replace('enhanced_inference_results.json', 'final_results.json')
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Final results saved to: {output_file}")
        return result
        
    except Exception as e:
        print(f"Error processing inference results: {e}")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process inference results with LLM')
    parser.add_argument('--inference_file', required=True, help='Path to enhanced inference results JSON')
    parser.add_argument('--output_file', help='Output file path')
    parser.add_argument('--api_key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    
    args = parser.parse_args()
    
    if args.api_key:
        os.environ['OPENAI_API_KEY'] = args.api_key
    
    result = process_inference_results(args.inference_file, args.output_file)
    
    if result:
        print(f"\n=== FINAL RESULTS ===")
        print(f"Signs: {result['signs']}")
        print(f"Sentence: {result['sentence']}")
        print(f"Emoji: {result['emoji']}")
        print(f"Sentiment: {result['sentiment']}") 