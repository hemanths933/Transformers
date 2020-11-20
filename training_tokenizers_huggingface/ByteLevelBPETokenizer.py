def train():
    from tokenizers import ByteLevelBPETokenizer

    '''
    Initialize a tokenizer
    '''
    tokenizer = ByteLevelBPETokenizer()

    '''
    Customize training

    Ex: Assuming I need <s> as StartOfSentence token and </s> as EndOfSentence token ,
    and <sep> token in case of seperation between subsentences etc. we specify the required special tokens. 
    These tokens are not broken into subword tokens by the tokenizer.

    '''
    paths = ['data/wiki_data.txt']
    tokenizer.train(files=paths, vocab_size=40000, min_frequency=2, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
        "<sep>"
    ])

    '''
    Save tokenizer
    '''
    tokenizer.save_model("./tok_checkpoints", "tokenizer_model")

def inference():

    from tokenizers import ByteLevelBPETokenizer
    from tokenizers.processors import BertProcessing

    '''
    initialize tokenizer with saved model files
    '''
    tokenizer = ByteLevelBPETokenizer(
        "./tok_checkpoints/tokenizer_model-vocab.json",
        "./tok_checkpoints/tokenizer_model-merges.txt",
    )



    '''
    optional step : preprocess the strings
    Ex: add <s> and </s> as BOS and EOS tokens to the string
        pad string to some max length and truncate string to some max length
    '''
    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )
    tokenizer.enable_padding(pad_token='<pad>', pad_id = tokenizer.get_vocab()['<pad>'], length=20)   
    tokenizer.enable_truncation(max_length=20)



    '''
    tokenize/encode strings
    '''
    input_ids = tokenizer.encode("Hello World, Whats up!!!").ids
    print("input ids", input_ids)
    tokens = tokenizer.encode("Hello World, Whats up!!!").tokens
    print("tokens", tokens)



    '''
    tokenize/encode batch of string
    '''
    batch_tokenized = tokenizer.encode_batch(["Hello World, Whats up!!!", "Whata whata wa wada wada"])
    input_ids = [i.ids for i in batch_tokenized]
    print("input ids", input_ids)
    tokens = [i.tokens for i in batch_tokenized]
    print("tokens", tokens)


if __name__ == '__main__':
    train()
    inference()
    

    

    