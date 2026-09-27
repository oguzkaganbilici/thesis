import os
import torch
import numpy as np
import itertools
from logger import setup_logger, log_training
from compute_metrics import evaluate_summary, evaluate_highlight

class Solver:
    def __init__(self, cfg, model, train_loader, val_loader, test_loader) -> None:
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.model = model.to(self.cfg.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr = self.cfg.learning_rate,
                                           weight_decay = self.cfg.weight_decay)

        if self.cfg.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max = self.cfg.num_epochs, eta_min=0)
        else:
            self.scheduler = None

        self.criterion = torch.nn.MSELoss()

        self.logger = setup_logger("solver", self.cfg.output_dir, overwrite=True)


    def train(self):
        best_model_score = -np.inf
        patience_counter = 0 # early stopping icin
        train_results = self.evaluate("train")
        val_results = self.evaluate("val")
        log_training(self.logger, train_results, val_results, 0)

        for epoch in range(1, self.cfg.num_epochs + 1):
            self.model.train()
            for batch in self.train_loader:

                visual = batch["visual_feat"].to(self.cfg.device)
                text = batch["text_feat"].to(self.cfg.device)
                audio = batch["audio_feat"].to(self.cfg.device)

                gt_score = batch["gt_score"].to(self.cfg.device)

                mask = batch["mask"].to(self.cfg.device)

                #forward
                output, _ = self.model(visual, text, audio, mask=mask) # output, attn_weights
                loss = self.criterion(output[mask], gt_score[mask])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            train_results = self.evaluate("train")
            val_results = self.evaluate("val")
            log_training(self.logger, train_results, val_results, epoch)

            model_score = val_results["ktau"] + val_results["srho"]

            if model_score > best_model_score:
                best_model_score = model_score
                patience_counter = 0
                best_model_ckpt = os.path.join(self.cfg.output_dir, "best_model_ckpt.pth")
                torch.save(self.model.state_dict(), best_model_ckpt)
                self.logger.info(f"\t# New best model at epoch {epoch}")

            else:
                patience_counter += 1

                if patience_counter >= self.cfg.patience:
                    self.logger.info(f"\t# Early stopping at epoch {epoch}")
                    break

    def evaluate(self, split="val"):
        self.model.eval()

        if split == "train":
            loader = list(itertools.islice(self.train_loader, len(self.val_loader)))
        elif split == "val":
            loader = self.val_loader
        else:
            loader = self.test_loader

        ktau_list, srho_list = [], []
        map50_list, map15_list = [], []
        loss_list = []

        with torch.no_grad():
            for batch in loader:
                visual = batch["visual_feat"].to(self.cfg.device)
                text = batch["text_feat"].to(self.cfg.device)
                audio = batch["audio_feat"].to(self.cfg.device)

                gt_score = batch["gt_score"].to(self.cfg.device)
                mask = batch["mask"].to(self.cfg.device)

                output, _ = self.model(visual, text, audio, mask=mask)
                loss = self.criterion(output[mask], gt_score[mask])

                loss_list.append(loss.item()) # loss bir tensor -> item() düz python sayısına cevirir.

                if output.dim() == 3:
                    output = output.squeeze(-1)

                pred_score = output.detach().cpu().numpy().tolist()
                gt_score = gt_score.detach().cpu().numpy().tolist()
                mask = mask.detach().cpu().numpy()

                ktau, srho = evaluate_summary(pred_score, gt_score, mask)
                map50, map15 = evaluate_highlight(pred_score, gt_score, mask)

                ktau_list.append(ktau)
                srho_list.append(srho)
                map50_list.append(map50)
                map15_list.append(map15)

        return {
            "ktau": np.mean(ktau_list),
            "srho": np.mean(srho_list),
            "map50": np.mean(map50_list),
            "map15": np.mean(map15_list),
            "loss": np.mean(loss_list),
            }

