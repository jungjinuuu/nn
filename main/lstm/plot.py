import matplotlib.pyplot as plt
import torch

def plot_loss(losses, filename: str = "loss_curve_lstm.png"):
    plt.figure(figsize = (6, 4))
    plt.plot(losses)
    plt.title("LSTM Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_prediction(y_true: torch.Tensor,
                    y_pred: torch.Tensor,
                    filename: str = "prediction_lstm.png"):
    y_true_np = y_true.squeeze(-1).detach().cpu().numpy()
    y_pred_np = y_pred.squeeze(-1).detach().cpu().numpy()

    plt.figure(figsize = (8, 4))
    plt.plot(y_true_np, label="True")
    plt.plot(y_pred_np, label="Predicted")
    plt.title("LSTM Next-Step Prediction (Test)")
    plt.xlabel("Time index (test range)")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_hidden(h_all: torch.Tensor,
                filename: str = "hidden_state_lstm.png"):
    """
    h_all: (B, T, H)
    """
    h = h_all[0]        # (T, H)
    h_dim0 = h[:, 0].detach().cpu().numpy()     # first hidden dimension

    plt.figure(figsize = (8, 4))
    plt.plot(h_dim0)
    plt.title("LSTM Hidden State (dim 0 of sample 0)")
    plt.xlabel("Time Step")
    plt.ylabel("Hidden Value")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()