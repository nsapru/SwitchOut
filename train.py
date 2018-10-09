#########################################################################
###########################   ADAPTED FROM   ############################
#######   http://nlp.seas.harvard.edu/2018/04/03/attention.html   #######
########  https://bastings.github.io/annotated_encoder_decoder/   #######
#########################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchtext import data, datasets
import numpy as np
import spacy
import math, copy, time
import matplotlib.pyplot as plt
import transformer



augmentation = False


seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)



##########################################################################
###########################   SwitchOut   ################################
######   Adapted from A.6 in https://arxiv.org/pdf/1808.07512.pdf   ######
##########################################################################

def switch_out(sents, tau, vocab_size,bos_id,eos_id,pad_id):
	"""
	Sample a batch of corrupted examples from sents.

	Args:
	sents: Tensor [batch_size, n_steps]. The input sentences.
	tau: Temperature.
	vocab_size: to create valid samples.
	Returns:
	sampled_sents: Tensor [batch_size, n_steps]. The corrupted sentences.

	"""
	mask = torch.eq(sents, bos_id) | torch.eq(sents, eos_id) | torch.eq(sents, pad_id)
	mask = mask.data.type('torch.ByteTensor') #converting to byte tensor for masked_fill in built function
	lengths = mask.float().sum(dim=1)
	batch_size, n_steps = sents.size()

	# first, sample the number of words to corrupt for each sentence
	logits = torch.arange(n_steps)
	logits = logits.mul_(-1).unsqueeze(0).expand_as(sents).contiguous().masked_fill_(mask, -float("inf"))
	logits = Variable(logits)
	probs = torch.nn.functional.softmax(logits.mul_(tau), dim=1)

	# finding corrupt sampels (most likely empty or 1 word) leading to zero prob
	for idx,prob in enumerate(probs.data):
		if torch.sum(prob)<= 0 and idx!=0:
			valid_ind = list(set(range(len(probs.data))))- list(set([idx]))
			for i in range(100):
				new_indx = random.choice(valid_list)
				if not torch.sum(probs.data[new_indx])<= 0:
					probs[idx] = probs[new_indx]
					break
				else:
					pass

	# still num_words probs fails likely due to corrupt input, therefore returning the whole original batch
	try:
		num_words = torch.distributions.Categorical(probs).sample()
	except:
		print ('Returning orignial batch!!!!!!')
		return sents

	corrupt_pos = num_words.data.float().div_(lengths).unsqueeze(1).expand_as(sents).contiguous().masked_fill_(mask, 0)

	corrupt_pos = torch.bernoulli(corrupt_pos, out=corrupt_pos).byte()
	total_words = int(corrupt_pos.sum())

	# sample the corrupted values, which will be added to sents
	corrupt_val = torch.LongTensor(total_words)
	corrupt_val = corrupt_val.random_(1, vocab_size)
	corrupts = torch.zeros(batch_size, n_steps).long()
	corrupts = corrupts.masked_scatter_(corrupt_pos, corrupt_val)
	corrupts = corrupts.cuda()
	sampled_sents = sents.add(Variable(corrupts)).remainder_(vocab_size)

	# converting sampled_sents into Variable before returning
	try:
		sampled_sents = Variable(sampled_sents)
	except:
		pass

	return sampled_sents



##########################################################################
#########################   DATA LOADING   ###############################
##########################################################################


print("Entering Data loop")

spacy_de = spacy.load('de')
spacy_en = spacy.load('en')

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = "<blank>"

print("Tokenizing Words....")
SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
TGT = data.Field(tokenize=tokenize_en, init_token = BOS_WORD, 
                 eos_token = EOS_WORD, pad_token=BLANK_WORD)

# Using a low max_len to minimize size of the dataset
MAX_LEN = 100

train, val, test = datasets.IWSLT.splits(
    exts=('.de', '.en'), fields=(SRC, TGT), 
    filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and 
        len(vars(x)['trg']) <= MAX_LEN)
MIN_FREQ = 2
SRC.build_vocab(train.src, min_freq=MIN_FREQ)
TGT.build_vocab(train.trg, min_freq=MIN_FREQ)



##########################################################################
#########################   DATASET SUMMARY   ############################
##########################################################################


def print_data_info(train_data, valid_data, test_data, src_field, trg_field):
    """ This prints some useful stuff about our data sets. """

    print("Data set sizes (number of sentence pairs):")
    print('train', len(train_data))
    print('valid', len(valid_data))
    print('test', len(test_data), "\n")

    print("First training example:")
    print("src:", " ".join(vars(train_data[0])['src']))
    print("trg:", " ".join(vars(train_data[0])['trg']), "\n")

    print("Most common words (src):")
    print("\n".join(["%10s %10d" % x for x in src_field.vocab.freqs.most_common(10)]), "\n")
    print("Most common words (trg):")
    print("\n".join(["%10s %10d" % x for x in trg_field.vocab.freqs.most_common(10)]), "\n")

    print("First 10 words (src):")
    print("\n".join(
        '%02d %s' % (i, t) for i, t in enumerate(src_field.vocab.itos[:10])), "\n")
    print("First 10 words (trg):")
    print("\n".join(
        '%02d %s' % (i, t) for i, t in enumerate(trg_field.vocab.itos[:10])), "\n")

    print("Number of German words (types):", len(src_field.vocab))
    print("Number of English words (types):", len(trg_field.vocab), "\n")


print_data_info(train, val, test, SRC, TGT)



##########################################################################
##################   OBJECT FOR BATCHES & MASKING   ######################
##########################################################################


class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            transformer.subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


##########################################################################
####################   CREATE BATCHES WITH TORCHTEXT   ###################
##########################################################################


global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)



##########################################################################
########################   ITERATOR FOR TORCHTEXT   ######################
##########################################################################


class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
            
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)



##########################################################################
#############################   TRAINING LOOP   ##########################
##########################################################################


def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):

        if (augmentation== False):
            print('Not Using SwitchOut....')
            out = model.forward(batch.src, batch.trg,
                            batch.src_mask, batch.trg_mask)

        elif (augmentation == True):
            print('Using SwitchOut....')

            tgt_pad_idx = TGT.vocab.stoi["<blank>"]
            tgt_eos_id = TGT.vocab.stoi['</s>']
            tgt_bos_id = TGT.vocab.stoi["<s>"]
            src_pad_idx = SRC.vocab.stoi["<blank>"]
            src_eos_id = SRC.vocab.stoi['</s>']
            src_bos_id = SRC.vocab.stoi["<s>"]

            bacth_switch_src = switch_out(batch.src,0.3,len(SRC.vocab),
                                          src_bos_id,src_eos_id,src_pad_idx)
            batch_switch_trg = switch_out(batch.trg,0.3,len(TGT.vocab),
                                          tgt_bos_id,tgt_eos_id,tgt_pad_idx)

            out = model.forward(bacth_switch_src, batch_switch_trg,
                                          batch.src_mask, batch.trg_mask)



        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0

        return total_loss / total_tokens


##########################################################################
###########################   MULTI-GPU MANAGER   ########################
##########################################################################


class MultiGPULossCompute:
    "A multi-gpu loss compute and train function."
    def __init__(self, generator, criterion, devices, opt=None, chunk_size=5):
        # Send out to different gpus.
        self.generator = generator
        self.criterion = nn.parallel.replicate(criterion, 
                                               devices=devices)
        self.opt = opt
        self.devices = devices
        self.chunk_size = chunk_size
        
    def __call__(self, out, targets, normalize):
        total = 0.0
        generator = nn.parallel.replicate(self.generator, 
                                                devices=self.devices)
        out_scatter = nn.parallel.scatter(out, 
                                          target_gpus=self.devices)
        out_grad = [[] for _ in out_scatter]
        targets = nn.parallel.scatter(targets, 
                                      target_gpus=self.devices)

        # Divide generating into chunks.
        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):
            # Predict distributions
            out_column = [[Variable(o[:, i:i+chunk_size].data, 
                                    requires_grad=self.opt is not None)] 
                           for o in out_scatter]
            gen = nn.parallel.parallel_apply(generator, out_column)

            # Compute loss. 
            y = [(g.contiguous().view(-1, g.size(-1)), 
                  t[:, i:i+chunk_size].contiguous().view(-1)) 
                 for g, t in zip(gen, targets)]
            loss = nn.parallel.parallel_apply(self.criterion, y)

            # Sum and normalize loss
            l = nn.parallel.gather(loss, 
                                   target_device=self.devices[0])
            l = l.sum()[0] / normalize
            total += l.data[0]

            # Backprop loss to output of transformer
            if self.opt is not None:
                l.backward()
                for j, l in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())

        # Backprop all loss through transformer.            
        if self.opt is not None:
            out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
            o1 = out
            o2 = nn.parallel.gather(out_grad, 
                                    target_device=self.devices[0])
            o1.backward(gradient=o2)
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return total * normalize


##########################################################################
###  model, criterion, optimizer, data iterators, and paralelization   ###
##########################################################################


devices = [0, 1, 2, 3]

print("Creating model, criterion, optimizer, data iterators, and paralelization")
print("Entering true training loop")
pad_idx = TGT.vocab.stoi["<blank>"]
print ("Building the Model")
model = transformer.make_model(len(SRC.vocab), len(TGT.vocab), N=6)
model.cuda()
criterion = transformer.LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
criterion.cuda()
BATCH_SIZE = 12000
train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=0,
                        repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                        batch_size_fn=batch_size_fn, train=True)
valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=0,
                        repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                        batch_size_fn=batch_size_fn, train=False)
model_par = nn.DataParallel(model, device_ids=devices)

model_opt = transformer.NoamOpt(model.src_embed[0].d_model, 1, 2000,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
for epoch in range(20):
    model_par.train()
    run_epoch((rebatch(pad_idx, b) for b in train_iter), 
              model_par, 
              MultiGPULossCompute(model.generator, criterion, 
                                  devices=devices, opt=model_opt))
    model_par.eval()
    loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter), 
                      model_par, 
                      MultiGPULossCompute(model.generator, criterion, 
                      devices=devices, opt=None))
    print('Val_loss: ',loss)

    
    
##########################################################################
########################  Adding test translator   #######################
##########################################################################

for i, batch in enumerate(valid_iter):
    src = batch.src.transpose(0, 1)[:1]
    src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
    out = transformer.greedy_decode(model, src, src_mask, 
                        max_len=60, start_symbol=TGT.vocab.stoi["<s>"])
    print("Translation:", end="\t")
    for i in range(1, out.size(1)):
        sym = TGT.vocab.itos[out[0, i]]
        if sym == "</s>": break
        print(sym, end =" ")
    print()
    print("Target:", end="\t")
    for i in range(1, batch.trg.size(0)):
        sym = TGT.vocab.itos[batch.trg.data[i, 0]]
        if sym == "</s>": break
        print(sym, end =" ")
    print()
    break
