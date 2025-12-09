import torch
import torch.nn as nn
import torch.optim as optim

from dataset import load_data
from model import LSTMPredictor
from plot import plot_loss, plot_prediction, plot_hidden


def train_lstm(
    num_epochs: int = 100,
    lr: float = 1e-3, 
    window_size: int = 20,
):

    torch.manual_seed(42)

    X_train, y_train, X_test, y_test = load_data(window_size=window_size)

    model = LSTMPredictor(input_size=1, hidden_size=32, num_layers=1)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        y_pred, h_all = model(X_train)
        loss = criterion(y_pred, y_train)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % 10 == 0:
            print(f"[Epoch {epoch:3d}] Loss = {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        y_pred_test, h_all_test = model(X_test)

    torch.save(model.state_dict(), "lstm_timeseries.pth")
    print("Saved model to lstm_timeseries.pth")

    plot_loss(losses, filename="loss_curve_lstm.png")
    plot_prediction(y_test, y_pred_test, filename="prediction_lstm.png")
    plot_hidden(h_all_test, filename="hidden_state_lstm.png")

    return model, (X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    train_lstm()