import torch
from transformers import pipeline, set_seed

@torch.no_grad()
def load_generation_model(model_name="gpt2"):
    """Loads the pretrained language model for text generation."""
    device = 0 if torch.cuda.is_available() else -1
    return pipeline('text-generation', model=model_name, device=device)

@torch.no_grad()
def load_sentiment_model():
    """Loads CardiffNLP Twitter-RoBERTa for nuanced 3-way sentiment classification."""
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        'sentiment-analysis',
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=device
    )

def generate_paired_completions(generator, template, demographics, max_length=50, seed=42):
    """
    Generates text completions for demographic subjects.
    Uses nucleus sampling (top_p) and temperature to prevent repetitive degeneration.
    """
    set_seed(seed)
    results = []
    
    for demo in demographics:
        prompt = template.format(demographic=demo)
        
        output = generator(
            prompt,
            max_length=max_length,
            num_return_sequences=1,
            truncation=True,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=generator.tokenizer.eos_token_id
        )
        
        completion = output[0]['generated_text']
        results.append({
            "Demographic": demo,
            "Prompt": prompt,
            "Completion": completion
        })
        
    return results

def analyze_bias(results, sentiment_analyzer):
    """
    Evaluates completions using RoBERTa.
    Labels are mapped to a continuous polarity scale:
      - POSITIVE:  +confidence score (0.0 to 1.0)
      - NEUTRAL:   0.0
      - NEGATIVE:  -confidence score (-1.0 to 0.0)
    """
    analysis_data = []
    
    for res in results:
        text = res['Completion']
        # RoBERTa maximum context is 512 tokens
        sentiment = sentiment_analyzer(text[:512])[0]
        
        raw_label = sentiment['label'].upper()
        conf = float(sentiment['score'])
        
        if raw_label == 'POSITIVE':
            score = conf
        elif raw_label == 'NEGATIVE':
            score = -conf
        else:
            raw_label = 'NEUTRAL'
            score = 0.0
            
        analysis_data.append({
            "Demographic": res['Demographic'],
            "Completion": res['Completion'],
            "Sentiment Label": raw_label,
            "Sentiment Score": round(score, 3)
        })
        
    return analysis_data