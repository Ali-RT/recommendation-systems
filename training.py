import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from config import PARAMS
from data import get_dataloaders, load_data
from model import get_model

# Data
data = load_data("ratings.csv")
train_loader, val_loader = get_dataloaders(data)


# Parameters
emb_size = PARAMS["emb_size"]
batch_size = PARAMS["batch_size"]
epochs = PARAMS["epochs"]
learning_rate = PARAMS["learning_rate"]

# Model
model = get_model()

# Loss
criterion = nn.MSELoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Data loader
# train_loader = DataLoader(train_ratings, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_ratings, batch_size=batch_size)

# Training loop
for epoch in range(epochs):
    model.train()
    for i, batch in enumerate(train_loader):
        # Get input batch
        user = Variable(batch[:, 0]).long()
        item = Variable(batch[:, 1]).long()
        rating = Variable(batch[:, 2]).float()

        # Forward pass
        prediction = model(user, item)
        loss = criterion(prediction, rating)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Eval
    model.eval()
    val_loss = 0
    for i, batch in enumerate(val_loader):
        with torch.no_grad():
            # Get input batch
            user = Variable(batch[:, 0]).long()
            item = Variable(batch[:, 1]).long()
            rating = Variable(batch[:, 2]).float()

            pred = model(user, item)
            loss = criterion(pred, rating)
        val_loss += loss.item()

    val_loss /= len(val_loader)
    # Print progress
    print(f"Epoch {epoch+1}: Train Loss {loss:.3f} | Val Loss {val_loss:.3f}")

    # Save model
    torch.save(model.state_dict(), f"{PARAMS['model_name']}_epoch{epoch}.pt")

print("Training complete!")
