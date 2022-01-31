import spacy
from spacy.language import Language
from rel_component.scripts.rel_pipe import make_relation_extractor, score_relations
from rel_component.scripts.rel_model import create_relation_model, create_classification_layer, create_instances
import random
import typer
from pathlib import Path
from spacy.tokens import DocBin, Doc
from spacy.training.example import Example

import json

tweets = []

for line in open("data/sample_tweets(1).txt"):
    temp = line.split('.')
    tweets += temp

print(tweets)

nlp = spacy.load("C:\\Users\\spate113\\Desktop\\Pycharm Projects\\NERAnnotation\\output\\model-last")

for doc in nlp.pipe(tweets, disable=["tagger"]):
    if doc.ents == ():
        continue
    print("Text:" + doc.text)
    print(f"spans: {[(e.text, e.label_) for e in doc.ents]}")

# nlp2 = spacy.load("rel_component/training_gpu/model-best")
# #
# for name, proc in nlp2.pipeline:
#     doc = proc(doc)
#
# print(doc._.rel.items())

# for value, rel_dict in doc._.rel.items():
#     for sent in doc.sents:
#         for e in sent.ents:
#             for b in sent.ents:
#                 if e.start == value[0] and b.start == value[1]:
#                     if rel_dict['INVOLVEDIN'] >= 0.9:
#                         print(f" entities: {e.text, b.text} --> predicted relation: {rel_dict}")


# for value, rel_dict in doc._.rel.items():
#     for e in doc.ents:
#         for b in doc.ents:
#             if e.start == value[0] and b.start == value[1]:
#                 if rel_dict['WONAGAINST'] >= 1.00e-10:
#                     print(f" entities: {e.text, b.text} --> predicted relation: {rel_dict}")
