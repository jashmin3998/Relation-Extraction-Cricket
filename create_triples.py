import spacy
import csv

tweets = []

# Reading tweets from txt file, separating it by full stops and storing it in tweets list
for line in open("data/sample_tweets.txt"):
    temp = line.split('.')
    tweets += temp

# loading NER model that was trained earlier
nlp = spacy.load("C:\\Users\\spate113\\Desktop\\Pycharm Projects\\NERAnnotation\\output\\model-last")

# List of entities and its relation with tweet
entity_list = [("COUNTRYTEAM", "hasContryTeam"),
               ("LEAGUETEAM", "hasLeagueTeam"),
               ("PLAYER", "hasPlayer"),
               ("BATSMAN", "hasBatsman"),
               ("TOURNAMENT", "hasTournament"),
               ("BOWLER", "hasBowler"),
               ("DATE", "hasDate"),
               ("RUN", "hasRuns"),
               ("EVENT", "hasEvent"),
               ("ACCOUNT", "hasAccount"),
               ("PERSON", "hasRelationWith"),
               ("LOCATION", "hasLocation"),
               ("CRICKET", "isRelatedTo")]

entity_dict = {"BOWLER": 0, "COUNTRYTEAM": 1, "LEAGUETEAM": 2, "PLAYER": 3, "BATSMAN": 4, "TOURNAMENT": 5,
               "DATE": 6, "RUN": 7, "EVENT": 8, "ACCOUNT": 9, "PERSON": 10, "LOCATION": 11, "CRICKET": 12}

# list for storing triples
temp = ["tweet", "hasBowler", "hasContryTeam", "hasLeagueTeam", "hasPlayer", "hasBatsman", "hasTournament", "hasDate",
        "hasRuns", "hasEvent", "hasAccount", "hasRelationWith",
        "hasLocation", "isRelatedTo"]

res = []
res.append(temp)
str = "www.mycricketweb.com/ontology#"
for doc in nlp.pipe(tweets, disable=["tagger"]):
    if doc.ents == ():
        continue
    text = doc.text.replace("\n", "")
    entities = [(e.text, e.label_) for e in doc.ents]

    for txt, ent in entities:
        ans = [""] * 14
        if entity_dict[ent] is not None:
            ans[0] = text
            ans[entity_dict[ent] + 1] = txt
            res.append(ans)
            print(ans)

print(res)

# open the file in the write mode
with open('output2.csv', 'w', encoding='UTF8', newline='') as f:
    # create the csv writer
    writer = csv.writer(f)
    for i in res:
        writer.writerow(i)

# Testing the triples form list

for i in res:
    print(i)
