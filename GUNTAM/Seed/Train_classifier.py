# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# """"""""""""""""""""""""""""""""""""""""" CLASSIFICATION T/F """""""""""""""""""""""""""""""""""""""""""""""""""""""""
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import torch
from GUNTAM.Transformer.Classifier_architecture import MLP_CE, train_loop_CE, icing_on_the_cake_CE
from GUNTAM.Seed.SeedTransformer import SeedTransformer
from GUNTAM.Seed.Config import SeedConfig
from GUNTAM.IO.prepare_classifier import seed_features_file_adjustment

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# """""""""""""""""""""""""""""""""""""""" TRANSFORMER LOADING """"""""""""""""""""""""""""""""

cfg = SeedConfig()
cfg.parse_args()
cfg.epoch_nb = 1

transformer = SeedTransformer(transformer_config=cfg.transformer_config, device_acc=cfg.device_acc, dtype=torch.float32)
transformer.to(cfg.device_acc)
transformer.load(path="transformer.pt", device=cfg.device_acc)


# """""""""""""""""""""""""""""""""""""" SEEDS FILE LOADING (build_seed_features_tensor) """"""""""""""""""""""""""""""""

seed_features = torch.load("seed_features.pt", weights_only=True)

# """""""""""""""""""""""""""""""""""""""""" TRAINING """""""""""""""""""""""""""""""""""""""""


def train_classifier(
    train_dataloader,
    input_shape,
    hidden_1,
    hidden_2,
    hidden_3,
    hidden_4,
    output_shape,
    p,
    n_epochs,
    lr,
    model_save: str,
    criterion=torch.nn.CrossEntropyLoss().to(device),
):

    model_CE = MLP_CE(
        input_shape=input_shape,
        hidden_1=hidden_1,
        hidden_2=hidden_2,
        hidden_3=hidden_3,
        hidden_4=hidden_4,
        output_shape=output_shape,
        p=p,
        activation=torch.nn.ReLU(),
    ).float()
    optimizer_CE = torch.optim.Adam(model_CE.parameters(), lr)
    model_CE = train_loop_CE(
        train_dataloader, model_CE, n_epochs, optimizer=optimizer_CE, criterion=criterion, lr=lr, device=device
    )
    model_CE = icing_on_the_cake_CE(train_dataloader, model_CE, n_epochs=5, lr=1e-3, device=device)
    torch.save(model_CE.state_dict(), model_save)


if __name__ == "__main__":

    train_dataloader, test_dataloader = seed_features_file_adjustment(data=seed_features, batch_size=1000)

    train_classifier(
        train_dataloader=train_dataloader,
        input_shape=28,
        hidden_1=512,
        hidden_2=256,
        hidden_3=128,
        hidden_4=64,
        output_shape=2,
        p=0,
        n_epochs=10,
        lr=1e-3,
        criterion="CE",
        model_save="classifier",
    )
