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
                "text": "Given a Chinese question, pick the most similar one from the list of 5 English questions."
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

def sim_check(eng_q, chi_q):
    client = OpenAI(api_key = os.environ['OPENAI_API_KEY'])
    user_prompt = f"# English Question:\n{eng_q}\n\n # Chinese Question:\n{chi_q}\n\n"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
            "role": "system",
            "content": [
                {
                "type": "text",
                "text": "Given an English question and a Chinese question, determine whether they are asking the same question."
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
                    "type": "boolean",
                    "description": "Whether the English question and Chinese question are asking the same question.",
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

def ir_top5_wrappper(question_detail, choices):
    error_count = 0
    while error_count < 3:
        try:
            gpt_pick = ir_top5(question_detail, choices)
        except:
            error_count += 1
            continue
    if error_count == 3:
        gpt_pick = "A"
        print("Error: GPT-4o-mini failed to respond. Defaulting to A.")
    gpt_pick_question = choices['ABCDE'.index(gpt_pick)]
    return gpt_pick, gpt_pick_question

def sim_check_wrapper(eng_q, chi_q):
    error_count = 0
    while error_count < 3:
        try:
            bl_match = sim_check(eng_q, chi_q)
        except:
            error_count += 1
            continue
    if error_count == 3:
        bl_match = False
        print("Error: GPT-4o-mini failed to respond. Defaulting to False.")
    return bl_match


reddit = pd.read_csv('../data/reddit_post.csv')
reddit = reddit[["message_id", "title", "message"]]
reddit.columns = ["message_id","question", "detail"]

with open('../data/emb/similarity.pkl', 'rb') as f:
    similarity = pickle.load(f)
    similarity = similarity.T

with open('../data/emb/zhihu_emb.pkl', 'rb') as f:
    raw = pickle.load(f)
    zh_sentences = raw[0]
    zh_emb = raw[1]
    zh_id = raw[2]

en_sentences = reddit.apply(lambda x: str(x['question'])+'\n'+str(x['detail']), axis=1).tolist()
en_id = reddit['message_id'].tolist()
num_en = len(en_sentences)
num_zh = len(zh_sentences)





# Example usage
results = []
top_k = 5
list_top_k = []
list_top_1 = []
list_match = []

# Convert to NumPy arrays for efficiency
zh_sentences = np.array(zh_sentences, dtype=object)
en_sentences = np.array(en_sentences, dtype=object)
en_id = np.array(en_id, dtype=object)
valid_mask = np.ones(num_en, dtype=bool)  # Track valid indices (True = selectable)

for i, zh_question in tqdm(enumerate(zh_sentences), total=num_zh):
    # Select only valid columns
    valid_similarity = np.where(valid_mask, similarity[i], -np.inf)  # Set invalid columns to -inf

    # Get top-k indices efficiently
    top_k_idx = np.argpartition(valid_similarity, -top_k)[-top_k:]  # O(n)
    top_k_idx = top_k_idx[np.argsort(valid_similarity[top_k_idx])[::-1]]  # O(k log k)

    top_k_sim = similarity[i][top_k_idx]
    selected_sentences = en_sentences[top_k_idx]
    selected_id = en_id[top_k_idx]

    list_top_k.append([(sim, idx) for sim, idx in zip(top_k_sim, selected_id)])

    gpt_pick, gpt_pick_question = ir_top5_wrappper(zh_question, selected_sentences)
    final_idx = top_k_idx['ABCDE'.index(gpt_pick)]
    list_top_1.append(en_id[final_idx])

    bl_match = sim_check_wrapper(gpt_pick_question, zh_question)
    list_match.append(bl_match)

    # Mark selected indices as unavailable (instead of deleting)
    if bl_match:
        valid_mask[final_idx] = False  # These sentences are no longer selectable


df_output = pd.DataFrame()

df_output['zh_question'] = zh_id
df_output['top_1'] = [x[0][1] for x in list_top_k]
df_output['top_1_sim'] = [x[0][0] for x in list_top_k]
df_output['top_2'] = [x[1][1] for x in list_top_k]
df_output['top_2_sim'] = [x[1][0] for x in list_top_k]
df_output['top_3'] = [x[2][1] for x in list_top_k]
df_output['top_3_sim'] = [x[2][0] for x in list_top_k]
df_output['top_4'] = [x[3][1] for x in list_top_k]
df_output['top_4_sim'] = [x[3][0] for x in list_top_k]
df_output['top_5'] = [x[4][1] for x in list_top_k]
df_output['top_5_sim'] = [x[4][0] for x in list_top_k]
df_output["gpt_pick"] = list_top_1
df_output["gpt_pick_question"] = list_match



df_output.to_csv('../data/r_matched_gpt_4o_mini.csv', index=False)