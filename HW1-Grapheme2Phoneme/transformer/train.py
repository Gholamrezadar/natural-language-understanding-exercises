#region imports
import os
import time
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import Levenshtein # pip install python-Levenshtein
#endregion

SOS_token = 0
EOS_token = 1

#region utilities
class Lang:
    def __init__(self, name):
        self.name = name
        self.char2index = {"SOS": 0, "EOS": 1}
        self.char2count = {}
        self.index2char = {0: "SOS", 1: "EOS"}
        self.n_chars = 2  # Count SOS and EOS

    def addWord(self, word):
        for char in word:
            self.addChar(char)

    def addChar(self, char):
        if char not in self.char2index:
            self.char2index[char] = self.n_chars
            self.index2char[self.n_chars] = char
            self.char2count[char] = 1
            self.n_chars += 1
        else:
            self.char2count[char] += 1

def readLangs(file_name="train_data.txt", portion=1.0):
    # Read the file and split into lines
    lines = open(file_name, encoding='utf-8').\
        read().strip().split('\n')
    
    lines = lines[:int(len(lines)*portion)]

    # Split every line into pairs
    pairs = [[s for s in l.split('\t')] for l in lines]

    # Make Lang instances
    input_lang = Lang("Grapheme")
    output_lang = Lang("Phoneme")

    return input_lang, output_lang, pairs

def loadData(file_name="train_data.txt", portion=1.0):
    input_lang, output_lang, pairs = readLangs(file_name=file_name, portion=portion)
    print("> Read %s sentence pairs" % len(pairs))
    for pair in pairs:
        input_lang.addWord(pair[0])
        output_lang.addWord(pair[1])
    print("\n> Counted chars:")
    print(input_lang.name, input_lang.n_chars)
    print(output_lang.name, output_lang.n_chars)
    return input_lang, output_lang, pairs

def phoneme_error_rate(p_seq1, p_seq2):
    p_vocab = set(p_seq1 + p_seq2)
    p2c = dict(zip(p_vocab, range(len(p_vocab))))
    c_seq1 = [chr(p2c[p]) for p in p_seq1]
    c_seq2 = [chr(p2c[p]) for p in p_seq2]
    return Levenshtein.distance(''.join(c_seq1),
                                ''.join(c_seq2)) / len(c_seq2)

def g2seq(s):
    return [g2idx['SOS']] + [g2idx[i] for i in s if i in g2idx.keys()] + [g2idx['EOS']]
    
def seq2g(s):
    return [idx2g[i] for i in s if idx2g[i]]

def p2seq(s):
    return [p2idx['SOS']] + [p2idx[i] for i in s if i in p2idx.keys()] + [p2idx['EOS']]

def seq2p(s):
    return [idx2p[i] for i in s]
# endregion

#region model
class TextLoader(torch.utils.data.Dataset):
    def __init__(self, path='train_data.txt'):
        self.x, self.y = [], []
        with open(path, 'r', encoding='utf-8') as f:
            data = f.read().strip().split('\n')
        for line in data:
            # print(line.split())
            x, y = line.split()
            self.x.append(g2seq(x))
            self.y.append(p2seq(y))

    def __getitem__(self, index):
        return (torch.LongTensor(self.x[index]), torch.LongTensor(self.y[index]))

    def __len__(self):
        return len(self.x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, intoken, outtoken, hidden, enc_layers=3, dec_layers=1, dropout=0.1):
        super(TransformerModel, self).__init__()
        nhead = hidden//64
        
        self.encoder = nn.Embedding(intoken, hidden)
        self.pos_encoder = PositionalEncoding(hidden, dropout)

        self.decoder = nn.Embedding(outtoken, hidden)
        self.pos_decoder = PositionalEncoding(hidden, dropout)

        self.transformer = nn.Transformer(d_model=hidden, nhead=nhead, num_encoder_layers=enc_layers, num_decoder_layers=dec_layers, dim_feedforward=hidden*4, dropout=dropout, activation='relu')
        self.fc_out = nn.Linear(hidden, outtoken)

        self.src_mask = None
        self.trg_mask = None
        self.memory_mask = None

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask==1, float('-inf'))
        return mask

    def make_len_mask(self, inp):
        return (inp == 0).transpose(0, 1)

    def forward(self, src, trg):
        if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
            self.trg_mask = self.generate_square_subsequent_mask(len(trg)).to(trg.device)

        src_pad_mask = self.make_len_mask(src)
        trg_pad_mask = self.make_len_mask(trg)

        src = self.encoder(src)
        src = self.pos_encoder(src)

        trg = self.decoder(trg)
        trg = self.pos_decoder(trg)

        output = self.transformer(src, trg, src_mask=self.src_mask, tgt_mask=self.trg_mask, memory_mask=self.memory_mask,
                                  src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=trg_pad_mask, memory_key_padding_mask=src_pad_mask)
        output = self.fc_out(output)

        return output

class TextCollate():
    def __call__(self, batch):
        max_x_len = max([i[0].size(0) for i in batch])
        x_padded = torch.LongTensor(max_x_len, len(batch))
        x_padded.zero_()

        max_y_len = max([i[1].size(0) for i in batch])
        y_padded = torch.LongTensor(max_y_len, len(batch))
        y_padded.zero_()

        for i in range(len(batch)):
            x = batch[i][0]
            x_padded[:x.size(0), i] = x
            y = batch[i][1]
            y_padded[:y.size(0), i] = y

        return x_padded, y_padded
#endregion

#region train and eval functions
def train(model, optimizer, criterion, iterator):
    
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        src, trg = batch
        src, trg = src.cuda(), trg.cuda()
        
        optimizer.zero_grad()
        output = model(src, trg[:-1,:])
        loss = criterion(output.transpose(0, 1).transpose(1, 2), trg[1:,:].transpose(0, 1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, criterion, iterator):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():    
        for i, batch in enumerate(iterator):
            src, trg = batch
            src, trg = src.cuda(), trg.cuda()

            output = model(src, trg[:-1,:])
            loss = criterion(output.transpose(0, 1).transpose(1, 2), trg[1:,:].transpose(0, 1))
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def validate(model, dataloader, show=10):
    model.eval()
    show_count = 0
    error_w = 0
    error_p = 0
    with torch.no_grad():
        for batch in dataloader:
            src, trg = batch
            src, trg = src.cuda(), trg.cuda()
            real_p = seq2p(trg.squeeze(1).tolist())
            real_g = seq2g(src.squeeze(1).tolist()[1:-1])

            memory = model.transformer.encoder(model.pos_encoder(model.encoder(src)))

            out_indexes = [p2idx['SOS'], ]

            for i in range(max_len):
                trg_tensor = torch.LongTensor(out_indexes).unsqueeze(1).to(device)

                output = model.fc_out(model.transformer.decoder(model.pos_decoder(model.decoder(trg_tensor)), memory))
                out_token = output.argmax(2)[-1].item()
                out_indexes.append(out_token)
                if out_token == p2idx['EOS']:
                    break

            out_p = seq2p(out_indexes)
            error_w += int(real_p != out_p)
            error_p += phoneme_error_rate(real_p, out_p)
            if show > show_count:
                show_count += 1
                print('Real g', ''.join(real_g))
                print('Real p', real_p)
                print('Pred p', out_p)
    return error_p/len(dataloader)*100, error_w/len(dataloader)*100

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
#endregion

if __name__ == "__main__":
    # Load train data
    grapheme_lang, phoneme_lang, pairs = loadData(file_name="train_data.txt")

    # initial settings
    random.seed(6969)
    torch.manual_seed(6969)
    torch.cuda.manual_seed(6969)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 1024

    # Fill Graphemes list from data
    graphemes = ['PAD', 'SOS']
    for grapheme in grapheme_lang.char2index:
        if grapheme == "EOS" or grapheme == "SOS":
            continue
        graphemes.append(grapheme)
    graphemes.append("EOS")

    # Fill Phonemes list from data
    phonemes = ['PAD', 'SOS']
    for phoneme in phoneme_lang.char2index:
        if phoneme == "EOS" or phoneme == "SOS":
            continue
        phonemes.append(phoneme)
    phonemes.append("EOS")

    # Create Grapheme to Index and Index to Grapheme dictionary for later use
    g2idx = {g: idx for idx, g in enumerate(graphemes)}
    idx2g = {idx: g for idx, g in enumerate(graphemes)}

    p2idx = {p: idx for idx, p in enumerate(phonemes)}
    idx2p = {idx: p for idx, p in enumerate(phonemes)}

    # Create dataset
    pin_memory = True
    num_workers = 2

    dataset = TextLoader()
    train_len = int(len(dataset) * 0.9)
    trainset, valset = torch.utils.data.random_split(dataset, [train_len, len(dataset) - train_len])

    collate_fn = TextCollate()

    train_loader = torch.utils.data.DataLoader(trainset, num_workers=num_workers, shuffle=True,
                            batch_size=batch_size, pin_memory=pin_memory,
                            drop_last=True, collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(valset, num_workers=num_workers, shuffle=False,
                            batch_size=batch_size, pin_memory=pin_memory,
                            drop_last=False, collate_fn=collate_fn)
    
    INPUT_DIM = len(graphemes)
    OUTPUT_DIM = len(phonemes)

    # Create the Transformer model
    model = TransformerModel(INPUT_DIM, OUTPUT_DIM, hidden=128, enc_layers=3, dec_layers=1).to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters())

    # Loss Function
    TRG_PAD_IDX = p2idx['PAD']
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    # Train
    N_EPOCHS = 200
    best_valid_loss = float('inf')

    # Train Loop
    print()
    for epoch in range(N_EPOCHS):
        print(f'Epoch: {epoch+1:02}')

        start_time = time.time()

        train_loss = train(model, optimizer, criterion, train_loader)
        valid_loss = evaluate(model, criterion, val_loader)

        epoch_mins, epoch_secs = epoch_time(start_time, time.time())

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'g2p_model_transformer.pt')

        print(f'> Time: {epoch_mins}m {epoch_secs}s | Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')

    print(">> best_valid_loss:", best_valid_loss)