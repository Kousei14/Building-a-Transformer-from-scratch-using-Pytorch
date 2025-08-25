from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "datasource": 'opus_books',
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "dataset/tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}" # opus_books_weights
    model_filename = f"{config['model_basename']}{epoch}.pt" # tmodel_{epoch}.pt
    return str(Path('.') / model_folder / model_filename) # opus_books_weights/tmodel_{epoch}.pt

def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}" # opus_books_weights
    model_filename = f"{config['model_basename']}*" # tmodel_{epoch}
    weights_files = list(Path(model_folder).glob(model_filename)) # [opus_books_weights/tmodel_{epoch01}.pt, opus_books_weights/tmodel_{epoch02}.pt, ...]
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1]) # gets the last tmodel_{epoch}.pt