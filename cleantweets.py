import pandas as pd
import re

def clean_text(text):
    #ensure it's a string
    text = str(text)

    #remove emails
    text = re.sub(r'\b[\w\-.]+?@\w+?\.\w{2,4}\b', ' ', text)
    
    #remove URLs
    text = re.sub(r'(http[s]?://\S+)|(\w+\.[A-Za-z]{2,4}\S*)', ' ', text)
    
    #remove phone numbers
    text = re.sub(r'\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b', ' ', text)

    #remove punctuation
    text = re.sub(r'[^\w\d\s]+', ' ', text)
    
    #limit to one space
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def main():
    input_file = 'Tweet.csv'
    output_file = 'Tweet_cleaned.csv'

    df = pd.read_csv(input_file)

    #drop unneccessary
    df.drop(columns=['writer', 'comment_num', 'retweet_num', 'like_num'], inplace=True)

    #clean tweets
    df['body'] = df['body'].apply(clean_text)

    df.to_csv(output_file, index=False)
    print(f"Cleaned file saved to: {output_file}")

if __name__ == "__main__":
    main()
