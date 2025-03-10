import os
import json
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from openai import OpenAI



def ir_top5(question, candidates):
        
    client = OpenAI(api_key = os.environ['OPENAI_API_KEY'])
    user_prompt = f"# Question:\n{question}\n\n # Candidates:\nA. {candidates[0]}\nB. {candidates[1]}\nC. {candidates[2]}\nD. {candidates[3]}\nE. {candidates[4]}\n\n"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
            "role": "system",
            "content": [
                {
                "type": "text",
                "text": "Given an English question, pick the most similar one from the list of 5 Chinese questions."
                }
            ]
            },
            {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": user_prompt
                    }
                ]
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
            "name": "similar_question_response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                "response": {
                    "type": "string",
                    "description": "The letter corresponding to the most similar question.",
                    "enum": [
                    "A",
                    "B",
                    "C",
                    "D",
                    "E"
                    ]
                }
                },
                "required": [
                "response"
                ],
                "additionalProperties": False
            }
            }
        },
        temperature=1,
        max_completion_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )
    
    return json.loads(response.choices[0].message.content)["response"]


reddit = pd.read_csv('../data/reddit_post.csv')
reddit = reddit[["message_id", "title", "message"]]
reddit.columns = ["message_id","question", "detail"]

with open('../data/emb/similarity.pkl', 'rb') as f:
    similarity = pickle.load(f)

with open('../data/emb/zhihu_emb.pkl', 'rb') as f:
    raw = pickle.load(f)
    zh_sentences = raw[0]
    zh_emb = raw[1]

def process_row(question_detail, choices):
    """Function to process each row"""
    gpt_pick = ir_top5(question_detail, choices)
    gpt_pick_question = choices['ABCDE'.index(gpt_pick)]
    return gpt_pick, gpt_pick_question



# Example usage
results = []
top_k = 5
list_top_k = []

num_samples = reddit.shape[0]
num_sentences = len(zh_sentences)

# Convert to NumPy arrays for efficiency
zh_sentences = np.array(zh_sentences, dtype=object)
valid_mask = np.ones(num_sentences, dtype=bool)  # Track valid indices (True = selectable)

for i, row in tqdm(enumerate(reddit.itertuples(index=False)), total=num_samples):
    # Select only valid columns
    valid_similarity = np.where(valid_mask, similarity[i], -np.inf)  # Set invalid columns to -inf

    # Get top-k indices efficiently
    top_k_idx = np.argpartition(valid_similarity, -top_k)[-top_k:]  # O(n)
    top_k_idx = top_k_idx[np.argsort(valid_similarity[top_k_idx])[::-1]]  # O(k log k)

    top_k_sim = similarity[i][top_k_idx]
    selected_sentences = zh_sentences[top_k_idx]

    list_top_k.append([(sim, sent) for sim, sent in zip(top_k_sim, selected_sentences)])

    question_detail = f"{row.question}\n{row.detail}"
    gpt_pick, gpt_pick_question = process_row(question_detail, selected_sentences)
    results.append((gpt_pick, gpt_pick_question))

    print(f"English: {row.question}")
    print(f'Chinese: {gpt_pick_question}')

    # Mark selected indices as unavailable (instead of deleting)
    selected_idx = top_k_idx['ABCDE'.index(gpt_pick)]
    valid_mask[selected_idx] = False  # These sentences are no longer selectable

    print(f"dropped: {zh_sentences[selected_idx]}")
    print("="*100)

reddit['top_1'] = [x[0][1] for x in list_top_k]
reddit['top_1_sim'] = [x[0][0] for x in list_top_k]
reddit['top_2'] = [x[1][1] for x in list_top_k]
reddit['top_2_sim'] = [x[1][0] for x in list_top_k]
reddit['top_3'] = [x[2][1] for x in list_top_k]
reddit['top_3_sim'] = [x[2][0] for x in list_top_k]
reddit['top_4'] = [x[3][1] for x in list_top_k]
reddit['top_4_sim'] = [x[3][0] for x in list_top_k]
reddit['top_5'] = [x[4][1] for x in list_top_k]
reddit['top_5_sim'] = [x[4][0] for x in list_top_k]

reddit["gpt_pick"], reddit["gpt_pick_question"] = zip(*results)



reddit.to_csv('../data/matched_gpt_4o_mini.csv', index=False)