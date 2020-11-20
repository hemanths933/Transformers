from torch.utils.data import Dataset
import torch
import torch.nn as nn
import os


class SplitReshapeTrainDataset(Dataset):
    def __init__(self, complex_sentences_file, simple_sentences_file):
        super(SplitReshapeTrainDataset, self).__init__()
        print("cwd", os.getcwd())
        self.complex_sentences = torch.load(complex_sentences_file)
        self.simple_sentences = torch.load(simple_sentences_file)
        self.device = torch.device('cpu')

    def __getitem__(self, index):
        #print("complex_sentences", self.complex_sentences)
        complex_sentence_input_ids = self.complex_sentences[index]
        #complex_sentence_attention_mask = self.complex_sentences['attention_mask'][index]
        simple_sentence_input_ids = self.simple_sentences[index]
        #simple_sentence_attention_mask = self.simple_sentences['attention_mask'][index]

        # print("complex_sentence_input_ids", complex_sentence_input_ids.shape)
        # print("complex_sentence_attention_mask", complex_sentence_attention_mask.shape)
        # print("simple_sentence_input_ids", simple_sentence_input_ids.shape)
        # print("simple_sentence_attention_mask", simple_sentence_attention_mask.shape)

        return complex_sentence_input_ids, simple_sentence_input_ids

    def __len__(self):
        return len(self.complex_sentences)

class SplitReshapeInferenceDataset(Dataset):
    def __init__(self, complex_sentences, tokenizer):
        super(SplitReshapeInferenceDataset, self).__init__()
        print("cwd", os.getcwd())
        self.device = torch.device('cpu')
        self.complex_sentences = complex_sentences
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        complex_sentence = self.complex_sentences[index]
        tokenized_complex_sentence = self.tokenizer.encode(complex_sentence)
        print("complex sentences", tokenized_complex_sentence.tokens)
        tokenized_complex_sentence_token_ids = torch.Tensor(tokenized_complex_sentence.ids)

        return tokenized_complex_sentence_token_ids

    def __len__(self):
        return len(self.complex_sentences)

if __name__ == '__main__':
    complex_sentences_file = '/home/ubuntu/Hemanth/split_and_reshape/data/tokenized_complex_sentences.pt'
    simple_sentences_file = '/home/ubuntu/Hemanth/split_and_reshape/data/tokenized_simple_sentences.pt'
    dataset = SplitReshapeTrainDataset(complex_sentences_file, simple_sentences_file)
    for i in dataset:
        print(i)
        break
