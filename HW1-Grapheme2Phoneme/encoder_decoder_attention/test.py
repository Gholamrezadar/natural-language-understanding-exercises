#region imports
import time
import math
import random
import copy

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from jiwer import cer, wer # pip install jiwer
#endregion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#region loading data
MAX_LENGTH = 34
SOS_token = 0
EOS_token = 1

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

def loadData(file_name="train_data.txt", portion=1.0):
    input_lang, output_lang, pairs = readLangs(file_name=file_name, portion=portion)
    # print("> Loaded %s Sentence Pairs" % len(pairs))
    for pair in pairs:
        input_lang.addWord(pair[0])
        output_lang.addWord(pair[1])
    # print("\n> Counted chars:")
    # print(input_lang.name, input_lang.n_chars)
    # print(output_lang.name, output_lang.n_chars)
    return input_lang, output_lang, pairs

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
#endregion

# region Encoder
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
#endregion

# region Attention Decoder
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
#endregion

#region training functions
def indexesFromWord(lang, word):
    return [lang.char2index[char] for char in word]

def tensorFromWord(lang, sentence):
    indexes = indexesFromWord(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair):
    input_tensor = tensorFromWord(input_lang, pair[0])
    target_tensor = tensorFromWord(output_lang, pair[1])
    return (input_tensor, target_tensor)

teacher_forcing_ratio = 0.5
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.001, encoder_optimizer=None, decoder_optimizer=None):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    last_loss = 0.0
    best_loss = 10000
    best_encoder = None
    best_decoder = None

    training_pairs = [tensorsFromPair(random.choice(pairs)) for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                    decoder, encoder_optimizer, decoder_optimizer, criterion)
        if loss<best_loss:
            best_loss = loss
            best_encoder = copy.deepcopy(encoder)
            best_decoder = copy.deepcopy(decoder)

        last_loss = loss
        print_loss_total += loss
        plot_loss_total += loss

        if print_every != 0:
            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f : ' % (timeSince(start, iter / n_iters),
                                            iter, iter / n_iters * 100, print_loss_avg), end="")
                evaluateRandomlyInline(encoder, decoder)
        if plot_every != 0:
            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

    if best_encoder != None:
        torch.save(best_encoder.state_dict(), "encoder.pt")
        torch.save(best_decoder.state_dict(), "decoder.pt")
    
    return last_loss
#endregion

#region eval functions
def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromWord(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                    encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2char[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

def evaluateRandomlyInline(encoder, decoder, n=4):
    for i in range(n):
        pair = random.choice(pairs)
        print(pair[0], end=":")
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ''.join(output_words)
        if i != n-1:
            print(output_sentence, end=" | ")
        else:
            print(output_sentence, end="")
    print()

def calculate_cer(encoder, decoder):
    a, b, test_pairs = loadData("test_data.txt")

    # len(test_pairs)
    test_graphemes = [i[0] for i in test_pairs]
    test_phonemes = [i[1] for i in test_pairs]

    eval_phoneme = []
    counter = 0
    for grapheme, phoneme in zip(test_graphemes, test_phonemes):
        counter += 1
        output_words, attentions = evaluate(encoder, decoder, grapheme)
        eval_phoneme.append("".join(output_words)[:-5])
        if counter % 1000 == 0:
            print(f"{counter} / {len(test_pairs)}")
    
    return cer(test_phonemes, eval_phoneme)

def calculate_wer(encoder, decoder):
    a, b, test_pairs = loadData("test_data.txt")

    test_graphemes = [i[0] for i in test_pairs]
    test_phonemes = [i[1] for i in test_pairs]

    eval_phoneme = []
    counter = 0
    for grapheme, phoneme in zip(test_graphemes, test_phonemes):
        counter += 1
        output_words, attentions = evaluate(encoder, decoder, grapheme)
        predicted_phoneme = "".join(output_words)[:-5]
        eval_phoneme.append(predicted_phoneme)
        if counter % 1000 == 0:
            print(f"{counter} / {len(test_pairs)}")
    # print(">", grapheme,"|", phoneme,"|", predicted_phoneme,"|", phoneme==predicted_phoneme)
    return wer(test_phonemes, eval_phoneme)
#endregion

if __name__ == "__main__":
    input_lang, output_lang, pairs = loadData(portion=1.0)

    hidden_size = 512
    learning_rate = 0.0003

    # Defining the model
    encoder = EncoderRNN(input_lang.n_chars, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_chars, dropout_p=0.1).to(device)

    # Loading pretrained weights
    encoder.load_state_dict(torch.load("encoder_best.pt"))
    decoder.load_state_dict(torch.load("decoder_best.pt"))

    #region jiwer cer, wer
    print("Calculating CER...")
    cer = calculate_cer(encoder, decoder)
    print()
    print("Calculating WER...")
    wer = calculate_wer(encoder, decoder)

    print()
    print("> CER: ", cer)
    print("> WER: ", wer)