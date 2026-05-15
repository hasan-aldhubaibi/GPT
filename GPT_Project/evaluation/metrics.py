import torch


def compute_loss(model, dataloader, device):
    model.eval()
    total_loss = 0
    count = 0

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            B, T, C = logits.shape

            loss = criterion(
                logits.view(B * T, C),
                y.view(B * T)
            )

            total_loss += loss.item()
            count += 1

    avg_loss = total_loss / count
    return avg_loss


def compute_perplexity(loss):
    return torch.exp(torch.tensor(loss)).item()