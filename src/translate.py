from pathlib import Path

from src.config import get_config, latest_weights_file_path, get_weights_file_path
from src.model import build_transformer
from src.dataset import BilingualDataset

from datasets import load_dataset

from tokenizers import Tokenizer

import torch
import sys

def translate(sentence: str):

    # Pre-processing - Prepare the SENTENCE PAIRS (source SENTENCE, target SENTENCE)
    # Note: If the sentence is a number, use it as an index to retrieve it from the test set
    label = ""
    if type(sentence) == int or sentence.isdigit():
        id = int(sentence)

        ds = load_dataset('json', data_files = 'train.json', split = 'all')
        ds = BilingualDataset(ds = ds, 
                              tokenizer_src = tokenizer_src, 
                              tokenizer_tgt = tokenizer_tgt, 
                              src_lang = config['lang_src'], 
                              tgt_lang = config['lang_tgt'], 
                              seq_len = config['seq_len'])
        sentence = ds[id]['src_text']
        label = ds[id]["tgt_text"]

    # Pre-processing - Define the TOKENIZERS (source, target), and seq_len
    config = get_config()
    tokenizer_src = Tokenizer.from_file(
        path = str(Path(config['tokenizer_file'].format(config['lang_src'])))
        ) # tokenizer_en.json
    tokenizer_tgt = Tokenizer.from_file(
        path = str(Path(config['tokenizer_file'].format(config['lang_tgt'])))
        ) # tokenizer_it.json
    seq_len = config['seq_len'] # seq_len

    # 1. Define the DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # 2. Define the MODEL
    model = build_transformer(src_vocab_size = tokenizer_src.get_vocab_size(), 
                              tgt_vocab_size = tokenizer_tgt.get_vocab_size(), 
                              src_seq_len = config["seq_len"], 
                              tgt_seq_len = config['seq_len'], 
                              d_model = config['d_model']).to(device)

    # 3. Load the PRETRAINED PARAMETERS (gets the last tmodel_{epoch}.pt)
    model_filename = latest_weights_file_path(config = config)
    state = torch.load(f = model_filename, 
                       weights_only = True)
    model.load_state_dict(state_dict = state['model_state_dict'])

    # 4. Forward Pass
    model.eval()
    with torch.no_grad():
        
        # ENCODER
        # 4.1 Tokenize the SENTENCE
        source = tokenizer_src.encode(sentence)
        # 4.2 Augment the SENTENCE - [SOS] + [source tokens] + [EOS] + [PADs]
        source = torch.cat([
            torch.tensor([tokenizer_src.token_to_id('[SOS]')], 
                         dtype = torch.int64), # start of sentence token
            torch.tensor(source.ids, 
                         dtype = torch.int64), # source tokens
            torch.tensor([tokenizer_src.token_to_id('[EOS]')], 
                         dtype = torch.int64), # end of sentence token
            torch.tensor([tokenizer_src.token_to_id('[PAD]')] * (seq_len - len(source.ids) - 2), 
                         dtype = torch.int64) # padding tokens (to complete context length)
                            ], dim = 0).to(device)
        source_mask = (source != tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)
        # 4.3 Encode the AUGMENTED SENTENCE + MASK
        encoder_output = model.encode(source, 
                                      source_mask)

        # Supplementary - Print the SOURCE SENTENCE, TARGET SENTENCE, PREDICTED SENTENCE
        if label != "": 
            print(f"{f'ID: ':>12}{id}") 
        print(f"{f'SOURCE: ':>12}{sentence}")
        if label != "": 
            print(f"{f'TARGET: ':>12}{label}") 
        print(f"{f'PREDICTED: ':>12}", end='')

        # DECODER
        # 4.4 Initialize the DECODER INPUT with the [SOS]
        decoder_input = torch.empty(1, 1).fill_(tokenizer_tgt.token_to_id('[SOS]')).type_as(source).to(device)
        # 4.5 Generate the TRANSLATION (word by word)
        while decoder_input.size(1) < seq_len:
            # 4.5.1 Build MASK for DECODER INPUT
            decoder_mask = torch.triu(input = torch.ones((1, decoder_input.size(1), decoder_input.size(1))), 
                                      diagonal = 1).type(torch.int).type_as(source_mask).to(device)
            # 4.5.2 Decode the [SOS] + MASK
            out = model.decode(
                encoder_output, 
                source_mask, 
                decoder_input, 
                decoder_mask
                )
            # 4.5.3 Project the NEXT TOKEN/s
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, 
                                     dim = 1)
            # 4.5.4 Augment the DECODER INPUT w/ the new tokens predicted
            decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], 
                                      dim = 1)

            # Supplementary - Print the TRANSLATED WORD
            print(f"{tokenizer_tgt.decode([next_word.item()])}", end = ' ')

            # 4.5.6 Break if [EOS] is predicted already
            if next_word == tokenizer_tgt.token_to_id('[EOS]'):
                break

    # 5. Convert DECODER INPUT IDs to TOKENS
    return tokenizer_tgt.decode(decoder_input[0].tolist())
    
sentence = input("Enter your text: ")
translate(sys.argv[1] if len(sys.argv) > 1 else sentence)