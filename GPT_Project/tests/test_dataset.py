# test_dataset.py (temporary)
from training.dataset import GPTDataset

dataset = GPTDataset("data/processed/train_ids.npy", block_size=8)

x, y = dataset[0]

print("x:", x)
print("y:", y)

print("Check shift:")
for i in range(5):
    print(x[i].item(), "->", y[i].item())