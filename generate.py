import torch
from dataset import load_data
from model import GPT


def load_model(model_path:str = "checkpoints/model.pt"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(model_path, map_location=device,weights_only=False)

    train_data, test_data, tokenizer = load_data()

    config = checkpoint['config']

    model = GPT(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dropout=0.0,
        block_size=config['block_size'],
    )

    model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)
    model.eval()

    return model,tokenizer,device

@torch.no_grad()
def generate(model,tokenizer,device,prompt:str,max_tokens:int = 500,temperature:float = 0.8):

    prompt_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(prompt_ids,dtype=torch.long).to(device).unsqueeze(0)

    output_ids = model.generate(input_ids,max_new_tokens=max_tokens,temperature=temperature)
    generated_text = tokenizer.decode(output_ids[0])

    return generated_text


if __name__=="__main__":
    model,tokenizer,device = load_model()
    while True:
        try:
            prompt = input("\nYour prompt: ")
            if prompt.lower() in ["exit","quit","q"]:
                print("Farewell")
                break
            if not prompt:
                continue
            generated_text = generate(model=model,
                                      tokenizer=tokenizer,
                                      device=device,
                                      prompt=prompt,
                                      max_tokens=500,
                                      temperature=0.8
            )

            print("\n" + generated_text)

        except KeyboardInterrupt:
            print("Farewell")
            break


