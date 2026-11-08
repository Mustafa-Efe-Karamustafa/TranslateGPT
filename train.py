import os
import time
import torch
from model import GPT
from dataset import load_data,get_bach

EMBEDDING_DIM = 384
NUM_HEADS = 6
NUM_LAYERS = 6
BLOCK_SIZE = 256
DROPOUT = 0.1

BATCH_SIZE = 64
MAX_ITERS = 5000
EVAL_INTERVAL = 500
LEARNING_RATE = 3e-4
WARMUP_ITERS = 100

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_DIR = 'checkpoints/model.pt'

def get_learning_rate(iteration):
    # 0-100 : 0 -> 0.0003
    if iteration < 100:
        return LEARNING_RATE * (iteration / WARMUP_ITERS)
    else:
        return LEARNING_RATE

@torch.no_grad()
def evaluate(model,train_data,test_data):
    model.eval()

    results = {}

    for name,data in [("train",train_data),("test",test_data)]:
        total_loss = 0

        for _ in range(100):
            x, y = get_bach(data,block_size=BLOCK_SIZE,batch_size=BATCH_SIZE)
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits,loss = model(x,y)
            total_loss += loss.item()

        results[name] = total_loss / 100
    model.train()
    return results


@torch.no_grad()
def generate_sample(model,tokenizer):
    model.eval()

    prompt = "ROMEO: "
    prompt_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(prompt_ids,dtype=torch.long).to(DEVICE).unsqueeze(0)

    output_ids = model.generate(input_ids,max_new_tokens=200,temperature=0.7)

    model.train()
    return tokenizer.decode(output_ids[0])


def train():
    train_data,test_data,tokenizer = load_data()
    vocab_size = tokenizer.vocab_size

    model = GPT(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        block_size=BLOCK_SIZE,
    )

    model = model.to(DEVICE)
    #model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE)
    os.makedirs("checkpoints",exist_ok=True)
    start_time = time.time()

    for iteration in range(MAX_ITERS):
        lr = get_learning_rate(iteration)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        x, y = get_bach(train_data,block_size=BLOCK_SIZE,batch_size=BATCH_SIZE)
        x, y = x.to(DEVICE), y.to(DEVICE)

        logits,loss = model(x,y)

        optimizer.zero_grad()
        loss.backward()

        #gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)

        optimizer.step()

        if iteration % EVAL_INTERVAL == 0 or iteration == MAX_ITERS - 1:
            losses = evaluate(model,train_data,test_data)
            elapsed_time = time.time() - start_time

            print(f"Iter {iteration}, "
                  f"Train loss: {losses['train']}, "
                  f"test_loss: {losses['test']}, ",
                  f"Learning rate: {lr},"
                  f"Time elapsed: {elapsed_time}")

            if iteration >0:
                print(generate_sample(model,tokenizer))

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "iteration": iteration,
        "test_loss": losses["test"],
        "config":{
            "vocab_size":vocab_size,
            "embedding_dim":EMBEDDING_DIM,
            "num_heads":NUM_HEADS,
            "num_layers":NUM_LAYERS,
            "dropout":DROPOUT,
            "block_size":BLOCK_SIZE,
        }
    }

    torch.save(checkpoint,CHECKPOINT_DIR)
    total_time = time.time() - start_time
    print(f"Total time: {total_time}")
    print(f"Final test loss: {losses['test']}")
    print(f"model saved to {CHECKPOINT_DIR}")

if __name__ == "__main__":
    train()
    '''
    train_data, test_data,tokenizer = load_data()
    gpt = GPT().to(DEVICE)
    print(generate_sample(gpt,tokenizer))
    '''