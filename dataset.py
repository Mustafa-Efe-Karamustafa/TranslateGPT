import torch
DATA_PATH = './data/data.txt'

class CharacterTokenizer:
    def __init__(self,text:str):

        self.chars = sorted(list(set(text)))

        self.vocab_size = len(self.chars)

        print(f"Chars: {self.chars}")
        print(f"Vocab: {self.vocab_size}")

        self.chars_to_id = {}

        for i,c in enumerate(self.chars):
            self.chars_to_id[c] = i

        self.id_to_char = {}
        for i,c in enumerate(self.chars):
            self.id_to_char[i] = c

    def encode(self, text : str)-> list:
        ls = []
        for c in text:
            ls.append(self.chars_to_id[c])
        return ls

    def decode(self, text : list)-> str:
        if isinstance(text,torch.Tensor):
            text = text.tolist()
        ls = []
        for c in text:
            ls.append(self.id_to_char[c])
        return ''.join(ls)

def load_data(train_size:float = 0.9):

    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        text = f.read()

    tokenizer = CharacterTokenizer(text)
    all_ids = tokenizer.encode(text)
    data = torch.tensor(all_ids,dtype=torch.long)

    split_idx = int(len(data) * train_size)

    train_data = data[:split_idx]
    test_data = data[split_idx:]

    return train_data, test_data,tokenizer

def get_batch(data:torch.Tensor,block_size:int,batch_size:int) -> torch.Tensor:
    max_start = len(data) - block_size -1
    positions = torch.randint(0,max_start,(batch_size,))


    x_list = []
    y_list = []

    for pos in positions:
        x_list.append(data[pos:pos+block_size])
        y_list.append(data[pos+1:pos+block_size+1])

    x = torch.stack(x_list)
    y = torch.stack(y_list)

    return x,y

if __name__ == "__main__":
    train_data, test_data,tokenizer = load_data()
    print(tokenizer.encode("efe"))
    print("efe\nbaba")



