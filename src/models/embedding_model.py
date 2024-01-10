# Author: Robert Guthrie
from src.api import WikipediaAPI
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import (rds_host, rds_dbname, rds_user, rds_password, rds_port)
from src.tables import ArticlesTable
import json

"""
Loading and Cleaning of DataFrame
- Current text_df has 3 cols: 
    id: PK, --> wiki key not cleaned
    text: str, --> Cleaned & not tokenized. Also no headings just straight up text.
    title: str --> Just the title of the article
    
- Next If: should likely be Tokenization method
- I believe there are diff ways that lead to diff results


- I think a focus on NER is better considering our ner is currently just string matching. 
"""
DATA = True

if DATA:
    emb_df = ArticlesTable(rds_dbname, rds_user, rds_password, rds_host, rds_port)
    json_obj = emb_df.get_all_data_json()
    emb_df.close_connection()
    with open('doccano_data.jsonl', 'w', encoding='utf-8') as file:
        for item in json_obj:
            item['text'] = item['text'].replace('\\', '')

            json_string = json.dumps(item, ensure_ascii=False)
            file.write(json_string + '\n')

    print("File 'doccano_data.jsonl' has been created.")

# torch.manual_seed(1)
#
#
# todo: Built proper embedding model
# word_to_ix = {"hello": 0, "world": 1}
# embeds = nn.Embedding(2, 5)  # 2 words in vocab, 5 dimensional embeddings
# lookup_tensor = torch.tensor([word_to_ix["hello"]], dtype=torch.long)
# hello_embed = embeds(lookup_tensor)
# print(hello_embed)
#
# CONTEXT_SIZE = 2
# EMBEDDING_DIM = 10
# # We will use Shakespeare Sonnet 2
# wiki_api = WikipediaAPI()
# TITLE = 'Normal_distribution'
# title, page_id, final_text = wiki_api.fetch_article_data(TITLE)
# print(final_text, '\n')
# test_sentence = remove_nested_curly_braces(final_text).split()
# print(test_sentence, '\n')
# # we should tokenize the input, but we will ignore that for now
# # build a list of tuples.
# # Each tuple is ([ word_i-CONTEXT_SIZE, ..., word_i-1 ], target word)
# ngrams = [
#     (
#         [test_sentence[i - j - 1] for j in range(CONTEXT_SIZE)],
#         test_sentence[i]
#     )
#     for i in range(CONTEXT_SIZE, len(test_sentence))
# ]
# # Print the first 3, just so you can see what they look like.
# print(ngrams[:3])
#
# vocab = set(test_sentence)
# word_to_ix = {word: i for i, word in enumerate(vocab)}
#
#
# class NGramLanguageModeler(nn.Module):
#
#     def __init__(self, vocab_size, embedding_dim, context_size):
#         super(NGramLanguageModeler, self).__init__()
#         self.embeddings = nn.Embedding(vocab_size, embedding_dim)
#         self.linear1 = nn.Linear(context_size * embedding_dim, 128)
#         self.linear2 = nn.Linear(128, vocab_size)
#
#     def forward(self, inputs):
#         embeds = self.embeddings(inputs).view((1, -1))
#         out = F.relu(self.linear1(embeds))
#         out = self.linear2(out)
#         log_probs = F.log_softmax(out, dim=1)
#         return log_probs
#
#
# losses = []
# loss_function = nn.NLLLoss()
# model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
# optimizer = optim.SGD(model.parameters(), lr=0.001)
#
# for epoch in range(10):
#     total_loss = 0
#     for context, target in ngrams:
#         # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
#         # into integer indices and wrap them in tensors)
#         context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
#
#         # Step 2. Recall that torch *accumulates* gradients. Before passing in a
#         # new instance, you need to zero out the gradients from the old
#         # instance
#         model.zero_grad()
#
#         # Step 3. Run the forward pass, getting log probabilities over next
#         # words
#         log_probs = model(context_idxs)
#
#         # Step 4. Compute your loss function. (Again, Torch wants the target
#         # word wrapped in a tensor)
#         loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))
#
#         # Step 5. Do the backward pass and update the gradient
#         loss.backward()
#         optimizer.step()
#
#         # Get the Python number from a 1-element Tensor by calling tensor.item()
#         total_loss += loss.item()
#     losses.append(total_loss)
# print(losses)  # The loss decreased every iteration over the training data!
#
# # To get the embedding of a particular word, e.g. "beauty"
# print(model.embeddings.weight[word_to_ix["beauty"]])
#
# CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
# raw_text = """We are about to study the idea of a computational process.
# Computational processes are abstract beings that inhabit computers.
# As they evolve, processes manipulate other abstract things called data.
# The evolution of a process is directed by a pattern of rules
# called a program. People create programs to direct processes. In effect,
# we conjure the spirits of the computer with our spells.""".split()
#
# # By deriving a set from `raw_text`, we deduplicate the array
# vocab = set(raw_text)
# vocab_size = len(vocab)
#
# word_to_ix = {word: i for i, word in enumerate(vocab)}
# data = []
# for i in range(CONTEXT_SIZE, len(raw_text) - CONTEXT_SIZE):
#     context = (
#             [raw_text[i - j - 1] for j in range(CONTEXT_SIZE)]
#             + [raw_text[i + j + 1] for j in range(CONTEXT_SIZE)]
#     )
#     target = raw_text[i]
#     data.append((context, target))
# print(data[:5])
#
# # class CBOW(nn.Module):
# #
# #     def __init__(self):
# #         pass
# #
# #     def forward(self, inputs):
# #         pass
# #
# # # Create your model and train. Here are some functions to help you make
# # # the data ready for use by your module.
# #
# #
# # def make_context_vector(context, word_to_ix):
# #     idxs = [word_to_ix[w] for w in context]
# #     return torch.tensor(idxs, dtype=torch.long)
# #
# #
# # make_context_vector(data[0][0], word_to_ix)  # example
