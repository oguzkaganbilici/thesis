import os
from config import Config
from torch.utils.data import DataLoader
from dataset import Dataset, CollateFn
from checkpoint_transfer import build_model_with_transfer
from solver import Solver

def main():
    cfg = Config(num_epochs=100, scheduler="cosine") # sadece bu 2 değeri override ediyoruz. Geri kalanlar aynı kalabilir.

    os.makedirs(cfg.output_dir, exist_ok=True)

    train_dataset = Dataset(cfg, "train")
    val_dataset = Dataset(cfg, "val")
    test_dataset = Dataset(cfg, "test")

    collate = CollateFn()

    train_loader = DataLoader(train_dataset, batch_size = cfg.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size = cfg.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size = cfg.batch_size, shuffle=False, collate_fn=collate)
    
    model = build_model_with_transfer(cfg)

    solver = Solver(cfg=cfg, model=model, 
                    train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)


    solver.train()


if __name__ == "__main__":
    main()

