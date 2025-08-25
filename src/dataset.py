import torch
from torch.utils.data import Dataset

class BilingualDataset(Dataset):

    def __init__(self, 
                 ds, 
                 tokenizer_src, 
                 tokenizer_tgt, 
                 src_lang, 
                 tgt_lang, 
                 seq_len):
        super().__init__()

        # seq_len: 350
        self.seq_len = seq_len

        # dataset: train and validation
        self.ds = ds

        # tokenizer for source corpus and target corpus. RETURN: Tokenizers
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt

        # src_lang: en, tgt_lang: it, RETURN: str(language)
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        # special tokens. RETURN: int(index of the special tokens in the vocab). e.g. for SOS = idx 2, EOS = idx 3, PAD = idx 1
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype = torch.int64) # start of sentence token
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype = torch.int64) # end of sentence token
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype = torch.int64) # pad token

    def __len__(self): # RETURN: the length of train or validation ds
        return len(self.ds)

    def __getitem__(self, idx):
        
        src_target_pair = self.ds[idx]

        src_text = src_target_pair['translation'][self.src_lang] # returns a single sentence
        tgt_text = src_target_pair['translation'][self.tgt_lang] # returns a single sentence

        # Transform the text into tokens. RETURN: List(ids)
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids # returns a list of ids of the tokenized src_text above
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids # returns a list of ids of the tokenized tgt_text above

        # Add <SOS>, <EOS> padding to each sentence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # We will add <SOS> and <EOS>
        # We will only add <SOS> only on the label
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # Make sure the num_padding_tokens is not negative. If it is, the sentence is too long
        # In essence, the length of sentence must <= be len(seq_len) - 2. -2 because we give way to the sos and eos tokens to be added
        # In this case, each sentence <= 360 - 2. In the dec_num_padding_tokens, <= 360 - 1 since we only add sos
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Add <SOS> and <EOS> token
        # as you can see, we add padding tokens to the src tensor to match the size of the seq_len by adding  the index of [PAD] tokens
        # we then stack the tensors horizontally. e.g. tensor([0, 1, 2,...])
        # see documentation for sample tensor output
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype = torch.int64),
            ],
            dim = 0,
        ) 

        # Add only <SOS> token
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype = torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype = torch.int64),
            ],
            dim = 0,
        )

        # Add only <EOS> token, uses the decoder list of indeces
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype = torch.int64),
            ],
            dim = 0,
        )

        # Double check the size of all the tensors (encoder and decoder input, label) to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len, seq_len)
            "label": label,  # (1, seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
    
def causal_mask(size):
    # as stated in the paper, the upper triangle is converted into zeros
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0