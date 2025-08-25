from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
# from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path

from datasets import load_dataset

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

def greedy_decode(model, 
                  source, 
                  source_mask, 
                  tokenizer_src, 
                  tokenizer_tgt, 
                  max_len, 
                  device):
    
    # Tokenize [SOS] and [EOS]
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Encode SENTENCE + MASK
    encoder_output = model.encode(source, source_mask)

    # Initialize the DECODER INPUT with the [SOS] 
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break
        # Build mask for DECODER INPUT
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        # Decode the [SOS] + MASK
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        # Project the NEXT TOKEN/ss
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        # Augment the DECODER INPUT w/ the new tokens predicted
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], 
            dim = 1
        )
        # Break if [EOS] is predicted already
        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, 
                   validation_ds, 
                   tokenizer_src, 
                   tokenizer_tgt, 
                   max_len, 
                   device, 
                   print_msg, 
                   global_step, 
                   writer, 
                   num_examples = 2):
    
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds: # batch_size = 1, to predict each sentence
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (batch, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (batch, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            # Generating tokens (returns the text output of the model)
            model_out = greedy_decode(model, 
                                      encoder_input, 
                                      encoder_mask, 
                                      tokenizer_src, 
                                      tokenizer_tgt, 
                                      max_len, 
                                      device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    if writer:
        # Evaluate the character error rate
        # --- Compute the char error rate 
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # --- Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # --- Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()

def get_all_sentences(ds, lang): # RETURN: Iterator
    # parses the sentences. e.g. In tha train.json, we call the "translation" key which outputs a dictionary
    # this dictionary contains our en and it sentences
    # we then call the "en" or "it" key to get our sentence
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang): # RETURN: Tokenizer
    # called separately for the source and target languages
    # e.g. "tokenizer_en.json". Creates a file.
    tokenizer_path = Path(config['tokenizer_file'].format(lang))

    # catch if the file is already created
    if not Path.exists(tokenizer_path):
        # WordLevel, Tokenizer
        tokenizer = Tokenizer(WordLevel(unk_token = "[UNK]"))

        # Whitespace
        tokenizer.pre_tokenizer = Whitespace()

        # WordLevelTrainer
        trainer = WordLevelTrainer(special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"], 
                                   min_frequency = 2)

        # fit()
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), 
                                      trainer = trainer)

        # save
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config): # RETURN: 2 Iterators (Dataset), 2 Tokenizers
    # It only has the train split, so we divide it overselves
    # ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')
    # I downloaded the dataset using google colab. I loaded the train.json file after
    ds_raw = load_dataset('json', 
                          data_files = 'dataset/train.json', 
                          split = 'train')

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src']) # creates the tokenizer_en.json, returns a tokenizer that will tokenize incoming texts
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt']) # creates the tokenizer_it.json, returns a tokenzier that will tokenize incoming texts

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw)) # 32332 * 0.9 = 29098
    val_ds_size = len(ds_raw) - train_ds_size # 32332 - 29099 = 3234
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size]) # splits the dataset according to the sizes

    # train dataset. RETURN: Dataset where we can access each index, and each index is a dictionary
    train_ds = BilingualDataset(train_ds_raw, 
                                tokenizer_src, 
                                tokenizer_tgt, 
                                config['lang_src'], 
                                config['lang_tgt'], 
                                config['seq_len'])
    # validation dataset
    val_ds = BilingualDataset(val_ds_raw, 
                              tokenizer_src, 
                              tokenizer_tgt, 
                              config['lang_src'], 
                              config['lang_tgt'], 
                              config['seq_len'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids # parses an english text / sentence and tokenizes it. RETURN: List(ids)
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids # parses an italian text / sentence and tokenizes it. RETURN: List(ids)
        max_len_src = max(max_len_src, len(src_ids)) # updates the max_len_src. After going all through the text / sentences, it will output the length of the longest
        max_len_tgt = max(max_len_tgt, len(tgt_ids)) # same with max_len_src

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    

    train_dataloader = DataLoader(
        train_ds, 
        batch_size = config['batch_size'], 
        shuffle = True
        ) # RETURN:  Iterator. In this case, it splits our dataset into 8 sentences per batch
    val_dataloader = DataLoader(
        val_ds, 
        batch_size = 1, 
        shuffle = True
        ) # RETURN: Iterator. Only one batch for the validation set

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, 
                              vocab_tgt_len, 
                              config["seq_len"], 
                              config['seq_len'], 
                              d_model=config['d_model'])
    return model

def train_model(config):

    # Supplementary - Make sure the weights folder exists (opus_books_weights)
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents = True, exist_ok = True)

    # get_ds() - Loads the 2 DATASETS and the 2 TOKENIZERS
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    # get_model() - Loads the TRANSFORMER MODEL
    model = get_model(config, 
                      tokenizer_src.get_vocab_size(), 
                      tokenizer_tgt.get_vocab_size()).to(device)
    
    # get_latest_weights_file_path() / latest_weight_file_path() - Loads PRE-TRAINED WEIGHTS (if it exists)
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    # Choose OPTIMIZER
    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], eps = 1e-9)

    # Choose LOSS FUNCTION
    loss_fn = nn.CrossEntropyLoss(ignore_index = tokenizer_src.token_to_id('[PAD]'), 
                                  label_smoothing = 0.1).to(device)

    # Choose DEVICE
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)

    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")

    device = torch.device(device)
    
    # Create BACKPROPAGATION BLOCK (Forward Pass, Compute Loss, Backward Pass)
    initial_epoch = 0
    global_step = 0

    for epoch in range(initial_epoch, config['num_epochs']):

        # Supplemental - Presets
        torch.cuda.empty_cache()
        model.train()

        # Supplemental - Loading bar
        batch_iterator = tqdm(train_dataloader, 
                              desc = f"Processing Epoch {epoch:02d}") # tqdm allows us to display progess to our iterable dataset

        # Supplemental - Initialize TensorBoard SummaryWriter
        writer = SummaryWriter(config['experiment_name'])

        for batch in batch_iterator: # For each SENTENCE in the batch_iterator
            encoder_input = batch['encoder_input'].to(device) # (batch, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (batch, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (batch, 1, seq_len, seq_len)

            # ENCODE each SOURCE and DECODE each TARGET
            encoder_output = model.encode(encoder_input, encoder_mask) # (batch, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (batch, seq_len, d_model)
            
            # 1. Forward Pass
            proj_output = model.project(decoder_output) # (batch, seq_len, vocab_size) -> y_pred
            label = batch['label'].to(device) # (batch, seq_len) -> y_true

            # 2. Compute Loss
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))

            # Supplemental - Update Loading bar with loss
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # 3. Backward pass
            optimizer.zero_grad(set_to_none = True)  # Clear previous gradients
            loss.backward()  # Compute new gradients
            optimizer.step()  # Update weights

            # Supplemental - Log the loss to TensorBoard
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            global_step += 1

        # Save the MODEL at the end of EVERY EPOCH (we'll have 20 model files in our folder)
        model_filename = get_weights_file_path(config, f"{epoch:02d}") # ./opus_books_weights/tmodel_{epoch}.pt
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }, 
            model_filename
            )
        print(f"Model saved to {model_filename}")
        
        # run_validation() - at the end of every epoch -> We have visibility of model predictions at every epoch
        run_validation(model, 
                       val_dataloader, 
                       tokenizer_src, 
                       tokenizer_tgt, 
                       config['seq_len'], 
                       device, lambda msg: batch_iterator.write(msg), 
                       global_step, 
                       writer)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)