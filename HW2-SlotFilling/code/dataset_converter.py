## Change the dataset format to a more standard one.
## Remove duplicate sentences.
## Extract unique intent_labels, slot_labels and words.
## Has 3 regions: Train Data, Test Data and Validation Data. try folding them to better understand the code.
## By Gholamreza Dar Fall 2022


#region Train Data

## Load data
print("Processing Train Data ...")
with open('old_format/train-en.conllu', 'r') as f:
    chunks = f.read().split('\n\n')

# Extract information from raw text
texts = []
intents = []
word_lists = []
tag_lists = []
for chunk in chunks:
    text = chunk[chunk.find("text: ")+6:chunk.find("\n# intent: ")]
    intent = chunk[chunk.find("# intent: ")+10:chunk.find("\n# slots: ")]
    if intent == "":
        continue
    word_list = []
    tag_list = []
    for line in chunk.split('\n'):
        if line.startswith('#'):
            continue
        if line == '':
            continue
        id, word, inten, tag = line.split('\t')
        if tag == 'NoLabel':
            tag = 'O'
        word_list.append(word)
        tag_list.append(tag)
    texts.append(text)
    intents.append(intent)
    word_lists.append(word_list)
    tag_lists.append(tag_list)

# find unique text ids
unique_ids = []
unique_texts = []
for idx, text in enumerate(texts):
    if text not in unique_texts:
        unique_texts.append(text)
        unique_ids.append(idx)
texts = unique_texts

# make intents, word_lists and tag_lists unique too
unique_intents = []
unique_word_lists = []
unique_tag_lists = []

for idx in unique_ids:
    unique_intents.append(intents[idx])
    unique_word_lists.append(word_lists[idx])
    unique_tag_lists.append(tag_lists[idx])

intents = unique_intents
word_lists = unique_word_lists
tag_lists = unique_tag_lists

## seq.in
with open('data/nlu/train/seq.in', 'w') as f:
    for word_list in word_lists:
        f.write(' '.join(word_list) + '\n')

## labels
with open('data/nlu/train/label', 'w') as f:
    for intent in intents:
        f.write(intent + '\n')

## seq.out
with open('data/nlu/train/seq.out', 'w') as f:
    for tag_list in tag_lists:
        f.write(' '.join(tag_list) + '\n')

## intent_label
with open('data/nlu/intent_label.txt', 'w') as f:
    unique_intents = list(set(intents))
    f.write("UNK\n")
    for intent in unique_intents:
        f.write(intent + '\n')

## slot_label
with open('data/nlu/slot_label.txt', 'w') as f:
    unique_tags = list(set([tag for tag_list in tag_lists for tag in tag_list]))
    f.write("PAD\n")
    f.write("UNK\n")
    f.write("<sos>\n")
    f.write("<eos>\n")
    for tag in unique_tags:
        f.write(tag + '\n')

## word_vocab
with open('data/nlu/word_vocab.txt', 'w') as f:
    unique_words = list(set([word for word_list in word_lists for word in word_list]))
    f.write("PAD\n")
    f.write("UNK\n")
    f.write("<sos>\n")
    f.write("<eos>\n")
    for word in unique_words:
        f.write(word + '\n')
    
#endregion

#region Valid Data
print("Processing Validation Data ...")
## Load data
with open('old_format/development-en.conllu', 'r') as f:
    chunks = f.read().split('\n\n')

# Extract information from raw text
texts = []
intents = []
word_lists = []
tag_lists = []
for chunk in chunks:
    text = chunk[chunk.find("text: ")+6:chunk.find("\n# intent: ")]
    intent = chunk[chunk.find("# intent: ")+10:chunk.find("\n# slots: ")]
    if intent == "":
        continue
    word_list = []
    tag_list = []
    for line in chunk.split('\n'):
        if line.startswith('#'):
            continue
        if line == '':
            continue
        id, word, inten, tag = line.split('\t')
        if tag == 'NoLabel':
            tag = 'O'
        word_list.append(word)
        tag_list.append(tag)
    texts.append(text)
    intents.append(intent)
    word_lists.append(word_list)
    tag_lists.append(tag_list)

# find unique text ids
unique_ids = []
unique_texts = []
for idx, text in enumerate(texts):
    if text not in unique_texts:
        unique_texts.append(text)
        unique_ids.append(idx)
texts = unique_texts

# make intents, word_lists and tag_lists unique too
unique_intents = []
unique_word_lists = []
unique_tag_lists = []

for idx in unique_ids:
    unique_intents.append(intents[idx])
    unique_word_lists.append(word_lists[idx])
    unique_tag_lists.append(tag_lists[idx])

intents = unique_intents
word_lists = unique_word_lists
tag_lists = unique_tag_lists

## seq.in
with open('data/nlu/dev/seq.in', 'w') as f:
    for word_list in word_lists:
        f.write(' '.join(word_list) + '\n')

## labels
with open('data/nlu/dev/label', 'w') as f:
    for intent in intents:
        f.write(intent + '\n')

## seq.out
with open('data/nlu/dev/seq.out', 'w') as f:
    for tag_list in tag_lists:
        f.write(' '.join(tag_list) + '\n')

## intent_label
current_intent_label = []
with open('data/nlu/intent_label.txt', 'r') as f:
    for line in f:
        current_intent_label.append(line.strip())

with open('data/nlu/intent_label.txt', 'a') as f:
    unique_intents = list(set(intents))
    for intent in unique_intents:
        if intent not in current_intent_label:
            f.write(intent + '\n')

## slot_label
current_slot_label = []
with open('data/nlu/slot_label.txt', 'r') as f:
    for line in f:
        current_slot_label.append(line.strip())

with open('data/nlu/slot_label.txt', 'a') as f:
    unique_tags = list(set([tag for tag_list in tag_lists for tag in tag_list]))
    for tag in unique_tags:
        if tag not in current_slot_label:
            f.write(tag + '\n')

## word_vocab
current_word_vocab = []
with open('data/nlu/word_vocab.txt', 'r') as f:
    for line in f:
        current_word_vocab.append(line.strip())

with open('data/nlu/word_vocab.txt', 'a') as f:
    unique_words = list(set([word for word_list in word_lists for word in word_list]))
    for word in unique_words:
        if word not in current_word_vocab:
            f.write(word + '\n')
#endregion

#region Test Data
print("Processing Test Data ...")
## Load data
with open('old_format/test-en.conllu', 'r') as f:
    chunks = f.read().split('\n\n')

# Extract information from raw text
texts = []
intents = []
word_lists = []
tag_lists = []
for chunk in chunks:
    text = chunk[chunk.find("text: ")+6:chunk.find("\n# intent: ")]
    intent = chunk[chunk.find("# intent: ")+10:chunk.find("\n# slots: ")]
    if intent == "":
        continue
    word_list = []
    tag_list = []
    for line in chunk.split('\n'):
        if line.startswith('#'):
            continue
        if line == '':
            continue
        id, word, inten, tag = line.split('\t')
        if tag == 'NoLabel':
            tag = 'O'
        word_list.append(word)
        tag_list.append(tag)
    texts.append(text)
    intents.append(intent)
    word_lists.append(word_list)
    tag_lists.append(tag_list)

# find unique text ids
unique_ids = []
unique_texts = []
for idx, text in enumerate(texts):
    if text not in unique_texts:
        unique_texts.append(text)
        unique_ids.append(idx)
texts = unique_texts

# make intents, word_lists and tag_lists unique too
unique_intents = []
unique_word_lists = []
unique_tag_lists = []

for idx in unique_ids:
    unique_intents.append(intents[idx])
    unique_word_lists.append(word_lists[idx])
    unique_tag_lists.append(tag_lists[idx])

intents = unique_intents
word_lists = unique_word_lists
tag_lists = unique_tag_lists

## seq.in
with open('data/nlu/test/seq.in', 'w') as f:
    for word_list in word_lists:
        f.write(' '.join(word_list) + '\n')

## labels
with open('data/nlu/test/label', 'w') as f:
    for intent in intents:
        f.write(intent + '\n')

## seq.out
with open('data/nlu/test/seq.out', 'w') as f:
    for tag_list in tag_lists:
        f.write(' '.join(tag_list) + '\n')

## intent_label
current_intent_label = []
with open('data/nlu/intent_label.txt', 'r') as f:
    for line in f:
        current_intent_label.append(line.strip())

with open('data/nlu/intent_label.txt', 'a') as f:
    unique_intents = list(set(intents))
    for intent in unique_intents:
        if intent not in current_intent_label:
            f.write(intent + '\n')

## slot_label
current_slot_label = []
with open('data/nlu/slot_label.txt', 'r') as f:
    for line in f:
        current_slot_label.append(line.strip())

with open('data/nlu/slot_label.txt', 'a') as f:
    unique_tags = list(set([tag for tag_list in tag_lists for tag in tag_list]))
    for tag in unique_tags:
        if tag not in current_slot_label:
            f.write(tag + '\n')

## word_vocab
current_word_vocab = []
with open('data/nlu/word_vocab.txt', 'r') as f:
    for line in f:
        current_word_vocab.append(line.strip())

with open('data/nlu/word_vocab.txt', 'a') as f:
    unique_words = list(set([word for word_list in word_lists for word in word_list]))
    for word in unique_words:
        if word not in current_word_vocab:
            f.write(word + '\n')
#endregion