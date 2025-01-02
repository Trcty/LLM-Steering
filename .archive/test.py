import transformers
import torch
from transformers import  AutoTokenizer
from datetime import datetime

model_id = "meta-llama/Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)


features = ["animal", "movement", "location", "color"]

def find_sequence(pipeline, tokenizer, question):
    sequences = pipeline(
        question,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=100,
    )
    print(sequences)

# Example sentences
sentences = [
    "The lion roared loudly as it ran across the savannah.",
    "The blue car sped down the highway, dodging traffic.",
    "A beautiful painting of a sunset hung on the wall.",
]

# Function to ask the model which features are present in a sentence
def identify_features_in_sentence(sentence, features):
    prompt = f"In the sentence: '{sentence}', which of these features are mentioned: {', '.join(features)}? Only feature name is needed."
    print(pipeline(prompt, max_length=100, num_return_sequences=1))
    response = pipeline(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
    return response

# Loop through sentences and check for features
for i, sentence in enumerate(sentences):
    prompt = f"In the sentence: '{sentence}', which of these features are mentioned: {', '.join(features)}? Please only provide feature name."
    find_sequence(pipeline, tokenizer, prompt)
    # response = identify_features_in_sentence(sentence, features)
    # print(f"Sentence {i+1}: {sentence}")
    # print(f"Model Response: {response}\n")