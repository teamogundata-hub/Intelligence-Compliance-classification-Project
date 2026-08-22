"""
ICC Training Module
====================
Complete training pipeline with domain pre-training, task fine-tuning,
early stopping, checkpointing, and multi-task loss balancing.

Author: Team Ogun — ICC Product
Date: August 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import logging
import os
import json
import time
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training hyperparameter configuration."""
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    early_stopping_patience: int = 3
    early_stopping_delta: float = 0.001
    checkpoint_every_n_steps: int = 500
    eval_every_n_steps: int = 100
    log_every_n_steps: int = 50
    task_weights: Dict[str, float] = None
    seed: int = 42

    def __post_init__(self):
        if self.task_weights is None:
            self.task_weights = {'kyc': 1.0, 'obligation': 1.0, 'risk': 1.0}


@dataclass
class TrainingMetrics:
    """Metrics recorded during training."""
    epoch: int
    step: int
    train_loss: float
    val_loss: float
    kyc_f1: float
    obligation_f1: float
    risk_f1: float
    overall_f1: float
    learning_rate: float
    elapsed_time: float


class EarlyStopping:
    """
    Early stopping handler based on validation F1 score.
    Supports multi-task early stopping with configurable patience.
    """

    def __init__(
        self,
        patience: int = 3,
        delta: float = 0.001,
        metric_name: str = 'overall_f1',
    ):
        self.patience = patience
        self.delta = delta
        self.metric_name = metric_name
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, metrics: dict) -> bool:
        score = metrics.get(self.metric_name, 0.0)

        if self.best_score is None:
            self.best_score = score
            return False

        if score < self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.should_stop


class CheckpointManager:
    """
    Manages model checkpointing during training.
    Saves best model and periodic checkpoints.
    """

    def __init__(self, output_dir: str, max_checkpoints: int = 5):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        metrics: dict,
    ) -> str:
        """Save a training checkpoint."""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        checkpoint_name = f"checkpoint_epoch{epoch}_step{step}_{timestamp}.pt"
        checkpoint_path = self.output_dir / checkpoint_name

        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
        }

        torch.save(checkpoint, checkpoint_path)
        self.checkpoints.append(str(checkpoint_path))

        # Remove old checkpoints
        while len(self.checkpoints) > self.max_checkpoints:
            old_path = self.checkpoints.pop(0)
            if os.path.exists(old_path):
                os.remove(old_path)

        logger.info(f"Saved checkpoint: {checkpoint_name}")
        return str(checkpoint_path)

    def save_best_model(self, model: nn.Module, tokenizer, metrics: dict):
        """Save the best model based on validation metrics."""
        best_path = self.output_dir / "best_model.pt"
        torch.save(model.state_dict(), best_path)

        # Also save tokenizer
        if tokenizer:
            tokenizer.save_pretrained(str(self.output_dir))

        # Save metrics
        metrics_path = self.output_dir / "best_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Saved best model with metrics: {metrics}")

    def load_checkpoint(self, checkpoint_path: str) -> dict:
        """Load a checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}, step {checkpoint['step']}")
        return checkpoint


class ICCTrainer:
    """
    Main trainer class for the ICC multi-head classifier.
    Implements multi-task training with task-specific loss weighting,
    gradient accumulation, and comprehensive evaluation.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        val_dataset,
        tokenizer,
        config: TrainingConfig,
        output_dir: str = "./checkpoints",
    ):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Set seed for reproducibility
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        self.model = model.to(self.device)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
        )
        self.tokenizer = tokenizer

        # Optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        total_steps = len(self.train_loader) * config.num_epochs // config.gradient_accumulation_steps
        warmup_steps = int(total_steps * config.warmup_ratio)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Loss functions
        self.kyc_criterion = nn.CrossEntropyLoss()
        self.obligation_criterion = nn.CrossEntropyLoss()
        self.risk_criterion = nn.CrossEntropyLoss()

        # Training utilities
        self.early_stopper = EarlyStopping(
            patience=config.early_stopping_patience,
            delta=config.early_stopping_delta,
        )
        self.checkpoint_manager = CheckpointManager(output_dir)
        self.training_history: List[TrainingMetrics] = []

    def _compute_loss(self, outputs: dict) -> torch.Tensor:
        """Compute weighted multi-task loss."""
        kyc_loss = outputs['kyc_loss']
        obligation_loss = outputs['obligation_loss']
        risk_loss = outputs['risk_loss']

        total_loss = (
            self.config.task_weights['kyc'] * kyc_loss +
            self.config.task_weights['obligation'] * obligation_loss +
            self.config.task_weights['risk'] * risk_loss
        )

        return total_loss

    def _train_step(self, batch: dict) -> dict:
        """Execute a single training step."""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        kyc_labels = batch['kyc_labels'].to(self.device)
        obligation_labels = batch['obligation_labels'].to(self.device)
        risk_labels = batch['risk_labels'].to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels_kyc=kyc_labels,
            labels_obligation=obligation_labels,
            labels_risk=risk_labels,
        )

        loss = self._compute_loss(outputs)
        loss = loss / self.config.gradient_accumulation_steps
        loss.backward()

        return {
            'total_loss': loss.item() * self.config.gradient_accumulation_steps,
            'kyc_loss': outputs['kyc_loss'].item(),
            'obligation_loss': outputs['obligation_loss'].item(),
            'risk_loss': outputs['risk_loss'].item(),
        }

    @torch.no_grad()
    def _evaluate(self) -> Dict[str, float]:
        """Run full evaluation on the validation set."""
        self.model.eval()

        all_preds = {'kyc': [], 'obligation': [], 'risk': []}
        all_labels = {'kyc': [], 'obligation': [], 'risk': []}
        val_losses = []

        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            kyc_labels = batch['kyc_labels'].to(self.device)
            obligation_labels = batch['obligation_labels'].to(self.device)
            risk_labels = batch['risk_labels'].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels_kyc=kyc_labels,
                labels_obligation=obligation_labels,
                labels_risk=risk_labels,
            )

            val_losses.append(outputs['total_loss'].item())

            all_preds['kyc'].extend(torch.argmax(outputs['kyc_logits'], dim=1).cpu().numpy())
            all_preds['obligation'].extend(torch.argmax(outputs['obligation_logits'], dim=1).cpu().numpy())
            all_preds['risk'].extend(torch.argmax(outputs['risk_logits'], dim=1).cpu().numpy())

            all_labels['kyc'].extend(kyc_labels.cpu().numpy())
            all_labels['obligation'].extend(obligation_labels.cpu().numpy())
            all_labels['risk'].extend(risk_labels.cpu().numpy())

        # Compute metrics
        metrics = {}
        for task in ['kyc', 'obligation', 'risk']:
            metrics[f'{task}_f1'] = f1_score(
                all_labels[task], all_preds[task], average='weighted', zero_division=0
            )
            metrics[f'{task}_precision'] = precision_score(
                all_labels[task], all_preds[task], average='weighted', zero_division=0
            )
            metrics[f'{task}_recall'] = recall_score(
                all_labels[task], all_preds[task], average='weighted', zero_division=0
            )

        metrics['overall_f1'] = (
            metrics['kyc_f1'] + metrics['obligation_f1'] + metrics['risk_f1']
        ) / 3
        metrics['val_loss'] = sum(val_losses) / len(val_losses) if val_losses else 0.0

        self.model.train()
        return metrics

    def train(self) -> List[TrainingMetrics]:
        """
        Execute the full training loop.

        Returns:
            List of TrainingMetrics for each epoch.
        """
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")

        self.model.train()
        global_step = 0
        start_time = time.time()

        for epoch in range(1, self.config.num_epochs + 1):
            epoch_start = time.time()
            epoch_losses = []

            for batch_idx, batch in enumerate(self.train_loader):
                loss_dict = self._train_step(batch)
                epoch_losses.append(loss_dict['total_loss'])

                # Gradient accumulation step
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                # Logging
                if global_step % self.config.log_every_n_steps == 0:
                    avg_loss = sum(epoch_losses[-self.config.log_every_n_steps:]) / self.config.log_every_n_steps
                    current_lr = self.scheduler.get_last_lr()[0]
                    logger.info(
                        f"Epoch {epoch} | Step {global_step} | "
                        f"Loss: {avg_loss:.4f} | LR: {current_lr:.2e}"
                    )

                # Evaluation
                if global_step % self.config.eval_every_n_steps == 0:
                    val_metrics = self._evaluate()
                    logger.info(
                        f"Validation | F1: {val_metrics['overall_f1']:.4f} | "
                        f"KYC: {val_metrics['kyc_f1']:.4f} | "
                        f"Obligation: {val_metrics['obligation_f1']:.4f} | "
                        f"Risk: {val_metrics['risk_f1']:.4f}"
                    )

                # Checkpointing
                if global_step % self.config.checkpoint_every_n_steps == 0:
                    self.checkpoint_manager.save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        step=global_step,
                        metrics=val_metrics,
                    )

            # End of epoch metrics
            epoch_time = time.time() - epoch_start
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            val_metrics = self._evaluate()

            epoch_metrics = TrainingMetrics(
                epoch=epoch,
                step=global_step,
                train_loss=avg_epoch_loss,
                val_loss=val_metrics['val_loss'],
                kyc_f1=val_metrics['kyc_f1'],
                obligation_f1=val_metrics['obligation_f1'],
                risk_f1=val_metrics['risk_f1'],
                overall_f1=val_metrics['overall_f1'],
                learning_rate=self.scheduler.get_last_lr()[0],
                elapsed_time=time.time() - start_time,
            )
            self.training_history.append(epoch_metrics)

            logger.info(
                f"Epoch {epoch} Complete | "
                f"Train Loss: {avg_epoch_loss:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Overall F1: {val_metrics['overall_f1']:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Save best model
            self.checkpoint_manager.save_best_model(
                self.model, self.tokenizer, val_metrics
            )

            # Early stopping check
            if self.early_stopper(val_metrics):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        logger.info(f"Training completed in {time.time() - start_time:.1f}s")
        return self.training_history

    def save_training_history(self, output_path: str):
        """Save training history to JSON."""
        history_data = [asdict(m) for m in self.training_history]
        with open(output_path, 'w') as f:
            json.dump(history_data, f, indent=2)
        logger.info(f"Training history saved to {output_path}")


# =============================================================================
# Domain Pre-Training (Masked Language Modeling)
# =============================================================================

class DomainPreTrainer:
    """
    Continues masked language modeling (MLM) pre-training on the
    CBN regulatory corpus to adapt the base model to Nigerian
    compliance terminology.
    """

    def __init__(
        self,
        model_name: str,
        tokenizer,
        corpus_path: str,
        output_dir: str = "./domain_pretrained",
        batch_size: int = 32,
        num_epochs: int = 3,
        learning_rate: float = 5e-5,
    ):
        from transformers import AutoModelForMaskedLM, LineByLineTextDataset, DataCollatorForLanguageModeling

        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.dataset = LineByLineTextDataset(
            tokenizer=tokenizer,
            file_path=corpus_path,
            block_size=128,
        )

        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.15,
        )

        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)

    def train(self):
        """Execute domain pre-training."""
        from transformers import get_linear_schedule_with_warmup

        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
        )

        total_steps = len(dataloader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps,
        )

        self.model.train()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                scheduler.step()
                self.optimizer.zero_grad()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            logger.info(f"Domain MLM Epoch {epoch + 1} | Loss: {avg_loss:.4f}")

        # Save domain-adapted model
        self.model.save_pretrained(str(self.output_dir))
        self.tokenizer.save_pretrained(str(self.output_dir))
        logger.info(f"Domain-adapted model saved to {self.output_dir}")
