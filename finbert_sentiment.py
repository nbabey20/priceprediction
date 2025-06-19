import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from tqdm import tqdm

def main():
    device = 0 if torch.cuda.is_available() else -1

    #finBERT model tokenizer
    model_name = "yiyanghkust/finbert-tone"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    finbert_pipe = pipeline(
        "sentiment-analysis", 
        model=model, 
        tokenizer=tokenizer, 
        device=device
    )

    input_file = "Tweet_cleaned_top_half.csv"
    output_file = "Tweet_labeled.csv"
    
    df = pd.read_csv(input_file)
    
    #body column to strings
    texts = df["body"].astype(str).tolist()

    batch_size = 64
    labels = []

    #sentimentally analyze and apply positive, negative or neutral
    for i in tqdm(range(0, len(texts), batch_size), desc="Classifying"):
        batch_texts = texts[i:i+batch_size]
        results = finbert_pipe(batch_texts, truncation=True)

        for r in results:
            labels.append(r["label"])


    df["Label"] = labels


    df.to_csv(output_file, index=False)
    print(f"Sentiment results saved to '{output_file}'.")

if __name__ == "__main__":
    main()
