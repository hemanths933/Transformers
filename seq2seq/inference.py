from utils import get_config, get_timestamp, log_metrics
from get_dataset import SplitReshapeInferenceDataset
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models.TransformerTranslationModel import TransformerModel

def initialize_model():

    config = get_config()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    print("device", device)

    '''create tokenizers'''

    tokenizer = ByteLevelBPETokenizer(
        "data/english_tokenizer-vocab.json",
        "data/english_tokenizer-merges.txt",
    )
    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )
    tokenizer.enable_padding(pad_token='[PAD]',length=config['max_len'])
    tokenizer.enable_truncation(max_length=config['max_len'])

    '''
    Create model
    '''
    vocab_size = len(tokenizer.get_vocab())
    print("tokenizer.vocab_size", vocab_size)
    model = TransformerModel(config['embedding_size'],
           vocab_size,
           vocab_size,
           config['src_pad_idx'],
           config['num_heads'],
           config['num_encoder_layers'],
           config['num_decoder_layers'],
           config['forward_expansion'],
           config['dropout'],
           config['max_len'],
           device)
    checkpoint = torch.load(config['pretrained_model'], map_location = device)
    model.load_state_dict(checkpoint['net'])
    model.eval()
    model = model.to(device)

    return config, model, tokenizer, device

config, model, tokenizer, device = initialize_model()

def split_and_rephrase_inference(complex_sentences):
    '''create dataset'''
    inference_dataset = SplitReshapeInferenceDataset(complex_sentences, tokenizer)
    prediction_dataloader = DataLoader(inference_dataset, batch_size=1, num_workers=0)

    model.eval()

    with torch.no_grad():
        for index, example in enumerate(prediction_dataloader):
            print("example", example)
            complex_sentences_input_ids = example.to(device).long().transpose(0, 1)
            
            outputs = [tokenizer.get_vocab()['<s>']]
            for i in range(config['max_len']):
                trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)
                with torch.no_grad():
                    output = model(complex_sentences_input_ids, trg_tensor)

                best_guess = output.argmax(2)[-1, :].item()
                outputs.append(best_guess)

                if best_guess == tokenizer.get_vocab()["</s>"]:
                    break
                
            print("outputs", outputs)

        return tokenizer.decode(outputs)

if __name__ == '__main__':
    print(split_and_rephrase_inference(['Enter username as Rohith and password as Password@123 then click login']))