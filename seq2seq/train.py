from utils import get_config, get_timestamp, log_metrics
from get_dataset import SplitReshapeTrainDataset
from transformers import DistilBertTokenizerFast
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.TransformerTranslationModel import TransformerModel
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import random
import time
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
import os

def initialize_model():

    config = get_config()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    print("device", device)

    '''
    Create dataloaders
    '''
    train_dataset = SplitReshapeTrainDataset(config['complex_sentences_file'], config['simple_sentences_file'])
    train_data, val_data = torch.utils.data.random_split(train_dataset, [round(config["train_data_percentage"] * len(train_dataset)), round(config["val_data_percentage"] * len(train_dataset))])

    train_dataloader = DataLoader(train_data, batch_size=config["batch_size"], num_workers=config["num_of_workers"], pin_memory=True)
    val_dataloader = DataLoader(val_data, batch_size=config["batch_size"], num_workers=config["num_of_workers"], pin_memory=True)

    '''
    create tokenizer
    '''
    tokenizer = ByteLevelBPETokenizer(
        "./data/english_tokenizer-vocab.json",
        "./data/english_tokenizer-merges.txt",
    )
    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )


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

    model.train()

    trainer = model.to(device)

    '''
    Create Optimizer
    '''
    loss_fun = nn.CrossEntropyLoss(ignore_index = config['src_pad_idx'])
    optimizer = optim.Adam(trainer.parameters(), lr = config["learning_rate"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)

    writer = SummaryWriter()

    return config, train_dataloader, val_dataloader, trainer, loss_fun, optimizer, writer, device, scheduler, tokenizer

config, train_dataloader, val_dataloader, trainer, loss_fun, optimizer, writer, device, scheduler, tokenizer = initialize_model()


def save_model(epoch, loss, experiment_id):

    save_dict = {}
    save_dict["net"] = trainer.state_dict()
    # save_dict['current_timestep'] = epoch + 1
    # save_dict['loss'] = loss
    # save_dict['optimizer'] = optimizer.state_dict()
    # save_dict['accuracy'] = accuracy
    # save_dict['max_desc_length'] = config["max_desc_length"]

    #timestamp = get_timestamp()
    

    '''
    save model dict
    '''
    state_dict_file_name = os.path.join(config["model_save_dir"], "split_and_rephrase" + \
                                    "#" + "loss" + '_' + str(loss) + \
                                    "#" + "epoch" + '_' + str(epoch) + \
                                    "#" + experiment_id + ".pt")
    torch.save(save_dict, state_dict_file_name)

    # '''
    # save pickle file
    # '''
    # pickle_file_name = os.path.join(config["model_save_dir"], "elementFinalizer_GGNN_multiclass" + \
    #                                 "#" + "acc" + '_' + str(accuracy) + \
    #                                 "#" + "epoch" + '_' + str(epoch) + \
    #                                 "#" + experiment_id + ".pkl")
    # with open(pickle_file_name, 'wb') as pkl_file:
    #     pickle.dump(trainer, pkl_file)

    return state_dict_file_name

def forwardbackwardpass(example):
            
    complex_sentences_input_ids = example[0].to(device).long().transpose(0, 1)
    #complex_sentences_attention_mask = example[1].to(device).long()
    #print("complex_sentences_input_ids", complex_sentences_input_ids.shape)
    simple_sentences_input_ids = example[1].to(device).long().transpose(0, 1)
    #print("simple_sentences_input_ids", simple_sentences_input_ids.shape)
    #simple_sentences_attention_mask = example[3].to(device).long()

    '''
    make optimizer zero grad
    '''
    optimizer.zero_grad()

    '''
    forward pass
    '''
    output = trainer(complex_sentences_input_ids, simple_sentences_input_ids[:-1])
    #print("output.shape", output.shape)
    seq_len, batch_size, _ = output.shape
    output = output.reshape(-1, output.shape[2])
    logits = output
    labels = simple_sentences_input_ids[1:].reshape(-1)
    #print("labels shape", labels.shape)

    



    loss = loss_fun(logits, labels)
    #print("loss is", loss)
    #backpass_time = time.time()
    #backward pass
    
    loss.backward()

    #apex backward pass
    # with amp.scale_loss(loss, optimizer) as scaled_loss:
    #     scaled_loss.backward()
    optimizer.step()
    
    #print("backpass_time", time.time() - backpass_time)

    logits = logits.reshape(batch_size, seq_len, -1)
    labels = labels.reshape(batch_size, seq_len)
    
    return loss, logits, labels


def forward_pass(example):
            
    complex_sentences_input_ids = example[0].to(device).long().transpose(0, 1)
    #complex_sentences_attention_mask = example[1].to(device).long()
    #print("complex_sentences_input_ids", complex_sentences_input_ids.shape)
    simple_sentences_input_ids = example[1].to(device).long().transpose(0, 1)
    #print("simple_sentences_input_ids", simple_sentences_input_ids.shape)
    #simple_sentences_attention_mask = example[3].to(device).long()

    '''
    make optimizer zero grad
    '''
    optimizer.zero_grad()

    '''
    forward pass
    '''
    output = trainer(complex_sentences_input_ids, simple_sentences_input_ids[:-1])
    #print("output.shape", output.shape)
    seq_len, batch_size, _ = output.shape
    output = output.reshape(-1, output.shape[2])
    logits = output
    labels = simple_sentences_input_ids[1:].reshape(-1)
    #print("labels shape", labels.shape)
    
    logits = logits.reshape(batch_size, seq_len, -1)
    labels = labels.reshape(batch_size, seq_len)

    return logits, labels

# def forward_pass(example):
#     #global trainer
#     #global optimizer

#     complex_sentences_input_ids = example[0].to(device).long().transpose(0, 1)
#     #complex_sentences_attention_mask = example[1].to(device).long()

#     outputs = [tokenizer.get_vocab()['<s>']]
#     for i in range(config['max_len']):
#         trg_tensor = torch.Tensor(outputs).unsqueeze(1).to(device).long()

#         with torch.no_grad():
#             output = trainer(complex_sentences_input_ids, trg_tensor)
        
#         best_guess = output.argmax(2)[-1, :].item()
#         outputs.append(best_guess)

#         if best_guess == tokenizer.get_vocab()['</s>']:
#             break

#     translated_sentence = [tokenizer.id_to_token(idx) for idx in outputs]
    
#     simple_sentences_input_ids = example[1].to(device).long().transpose(0, 1)
#     #simple_sentences_attention_mask = example[3].to(device).long()

#     '''
#     forward pass
#     '''
    
#     output = trainer(complex_sentences_input_ids, simple_sentences_input_ids[:-1])
#     logits = gcn_out
    
#     return logits, labels
    
def write_graph():
    #global train_dataloader
    #global trainer
    #global writer

    '''
    Write Graph
    '''
    for example in train_dataloader:
        complex_sentences_input_ids = example[0].to(device).long().transpose(0, 1)
        #print("complex_sentences_input_ids max", torch.max(complex_sentences_input_ids))
        #print("complex_sentences_input_ids min", torch.min(complex_sentences_input_ids))
        #print("complex_sentences_input_ids", complex_sentences_input_ids.shape)
        # complex_sentences_attention_mask = example[1].to(device).float()
        # print("complex_sentences_attention_mask", complex_sentences_attention_mask.dtype)

        simple_sentences_input_ids = example[1].to(device).long().transpose(0, 1)
        #print("simple_sentences_input_ids max", torch.max(simple_sentences_input_ids))
        #print("simple_sentences_input_ids max", torch.min(simple_sentences_input_ids))
        #print("simple_sentences_input_ids", simple_sentences_input_ids[:-1].shape)
        #simple_sentences_attention_mask = example[3].to(device)
        
        writer.add_graph(trainer, (complex_sentences_input_ids, simple_sentences_input_ids[:-1]))
        break


def train_one_epoch(epoch, experiment_id):
    #global writer
    #global train_dataloader

    #print("Epoch is ", epoch)
    epoch_start_time = time.time()
        
    running_loss = 0
    running_labels = []
    running_predicted = []
    running_samples = 0
    
    for timestep, example in tqdm(enumerate(train_dataloader)):
        #print("timestep", timestep)
        loss, pred, labels = forwardbackwardpass(example)
        #print("pred shape", pred.shape)
        #print("pred max shape", pred.argmax(2).shape)
        
        '''convert tokens to string to predict blue score'''
        

        if timestep % 5 == 0:
            scheduler.step()
        #print("predicted", predicted.shape)
        #print("labels", labels.shape)

        running_loss += loss.item()
        running_samples += example[0].shape[0]
        
    blue, loss = log_metrics(writer, epoch, running_predicted, running_labels, 'train', config, running_loss, running_samples)
    trainer.eval()
    with torch.no_grad():
        run_val(epoch)

    if epoch%config['model_save_frequency_in_epochs'] == 0:
        save_model(epoch, loss, experiment_id)
    print("time taken for the epoch is ", time.time() - epoch_start_time)

def run_val(epoch):
    #global val_dataloader
    #global writer

    running_labels = []
    running_predicted = []
    for timestep, example in tqdm(enumerate(val_dataloader)):
        pred, labels = forward_pass(example)
        #predicted = torch.round(pred)
        #print("predicted", predicted.shape)
        #print("labels", labels.shape)
        # running_labels += labels.view(-1).cpu().detach().tolist()
        # running_predicted += predicted.view(-1).cpu().detach().tolist()

    log_metrics(writer, epoch, running_predicted, running_labels, 'val', config)
    

def main():
    #global config

    #save label encoders
    experiment_id = str(random.randrange(0,1000000))
    label_encoder_prefix = experiment_id + '_'

    '''
    write graph
    '''
    #write_graph()

    '''
    Train Loop
    '''

    for epoch in tqdm(range(config["num_of_epochs"])):
        train_one_epoch(epoch, experiment_id)
        print("learning rate", scheduler.get_lr())
    
if __name__ == '__main__':
    main()