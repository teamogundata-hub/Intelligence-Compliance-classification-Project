"""
ICC Multi-Head Transformer Architecture
=========================================
A transformer-based architecture with multiple classification heads
for KYC tiering, obligation mapping, and risk flagging.

Supports base models: Legal-BERT, FinBERT, and custom embeddings.

Author: Team Ogun — ICC Product
Date: August 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoTokenizer
from typing import Optional, Dict, List
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AttentionPooling(nn.Module):
    """
    Attention-based pooling layer for combining token embeddings
    into a single document representation.
    """

    def __init__(self, hidden_size: int):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute attention-weighted sum of hidden states.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]

        Returns:
            Pooled representation: [batch_size, hidden_size]
        """
        # Compute attention scores
        attention_scores = self.attention(hidden_states).squeeze(-1)  # [batch, seq_len]

        # Mask padding tokens
        attention_scores = attention_scores.masked_fill(
            attention_mask == 0, -1e9
        )

        # Softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch, seq_len]

        # Weighted sum
        pooled = torch.bmm(
            attention_weights.unsqueeze(1),  # [batch, 1, seq_len]
            hidden_states  # [batch, seq_len, hidden_size]
        ).squeeze(1)  # [batch, hidden_size]

        return pooled


class TaskSpecificHead(nn.Module):
    """
    A task-specific classification head with optional layer normalization
    and configurable depth.
    """

    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_layer_norm: bool = True,
    ):
        super(TaskSpecificHead, self).__init__()

        layers = []
        for i in range(num_layers):
            in_features = hidden_size if i == 0 else hidden_size
            layers.append(nn.Linear(in_features, hidden_size))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))

        self.encoder = nn.ModuleList(layers)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.encoder:
            x = layer(x)
        return self.classifier(x)


class NigerianEntityEmbedding(nn.Module):
    """
    Custom embedding layer for Nigerian entity features (BVN presence,
    NIN presence, tier indicators) that can be concatenated with
    transformer outputs.
    """

    def __init__(self, num_entity_features: int = 4, embedding_dim: int = 64):
        super(NigerianEntityEmbedding, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_entity_features,
            embedding_dim=embedding_dim,
        )
        self.fc = nn.Linear(embedding_dim * num_entity_features, 128)
        self.norm = nn.LayerNorm(128)

    def forward(self, entity_features: torch.Tensor) -> torch.Tensor:
        """
        Process entity features into dense embeddings.

        Args:
            entity_features: [batch_size, num_entity_features] integer tensor

        Returns:
            Dense entity embedding: [batch_size, 128]
        """
        embeddings = self.embedding(entity_features)  # [batch, num_features, dim]
        flattened = embeddings.view(embeddings.size(0), -1)  # [batch, num_features * dim]
        output = self.fc(flattened)
        output = self.norm(output)
        return output


class ICCMultiHeadClassifier(nn.Module):
    """
    Main ICC classifier with three task-specific heads:
    1. KYC Tier Classification (3 classes)
    2. Obligation Mapping (50+ CBN obligation categories)
    3. Risk Flagging (binary: suspicious vs normal, or multi-class)

    Architecture:
        Input IDs + Attention Mask
                |
        Transformer Encoder (Legal-BERT / FinBERT)
                |
        [CLS] Pooling or Attention Pooling
                |
        ┌───────┼───────┐
        │       │       │
    KYC Head  Obligation Head  Risk Head
    """

    SUPPORTED_BASE_MODELS = {
        'legal-bert': 'nlpaueb/legal-bert-base-uncased',
        'finbert': 'ProsusAI/finbert',
        'bert-base': 'bert-base-uncased',
        'roberta-base': 'roberta-base',
    }

    def __init__(
        self,
        base_model_name: str = 'nlpaueb/legal-bert-base-uncased',
        num_kyc_classes: int = 3,
        num_obligation_classes: int = 15,
        num_risk_classes: int = 2,
        dropout: float = 0.3,
        pooling_strategy: str = 'cls',
        use_entity_features: bool = False,
        freeze_base_layers: int = 4,
    ):
        """
        Initialize the ICC multi-head classifier.

        Args:
            base_model_name: Name or path of the base transformer model.
            num_kyc_classes: Number of KYC tier classes.
            num_obligation_classes: Number of obligation categories.
            num_risk_classes: Number of risk flag classes.
            dropout: Dropout probability.
            pooling_strategy: 'cls' for CLS token, 'attention' for attention pooling.
            use_entity_features: Whether to use Nigerian entity features.
            freeze_base_layers: Number of base model layers to freeze.
        """
        super(ICCMultiHeadClassifier, self).__init__()

        self.config = AutoConfig.from_pretrained(base_model_name)
        self.bert = AutoModel.from_pretrained(base_model_name)
        self.pooling_strategy = pooling_strategy
        self.use_entity_features = use_entity_features

        hidden_size = self.config.hidden_size

        # Optional attention pooling
        if pooling_strategy == 'attention':
            self.attention_pooling = AttentionPooling(hidden_size)

        # Optional entity feature integration
        if use_entity_features:
            self.entity_embedding = NigerianEntityEmbedding(
                num_entity_features=4,
                embedding_dim=64
            )
            effective_hidden_size = hidden_size + 128
        else:
            effective_hidden_size = hidden_size

        # Task-specific classification heads
        self.kyc_head = TaskSpecificHead(
            hidden_size=effective_hidden_size,
            num_classes=num_kyc_classes,
            num_layers=2,
            dropout=dropout,
        )

        self.obligation_head = TaskSpecificHead(
            hidden_size=effective_hidden_size,
            num_classes=num_obligation_classes,
            num_layers=2,
            dropout=dropout,
        )

        self.risk_head = TaskSpecificHead(
            hidden_size=effective_hidden_size,
            num_classes=num_risk_classes,
            num_layers=2,
            dropout=dropout,
        )

        # Freeze base layers
        self._freeze_base_layers(freeze_base_layers)

    def _freeze_base_layers(self, num_layers: int):
        """Freeze the first N layers of the base model for initial training."""
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in self.bert.encoder.layer[:num_layers]:
            for param in layer.parameters():
                param.requires_grad = False
        logger.info(f"Froze first {num_layers} layers of base model")

    def _pool_hidden_states(
        self,
        outputs: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Extract pooled representation from transformer outputs."""
        if self.pooling_strategy == 'cls':
            return outputs[:, 0, :]  # [CLS] token
        elif self.pooling_strategy == 'attention':
            return self.attention_pooling(outputs, attention_mask)
        elif self.pooling_strategy == 'mean':
            # Masked mean pooling
            mask = attention_mask.unsqueeze(-1).float()
            return torch.sum(outputs * mask, dim=1) / torch.clamp(torch.sum(mask, dim=1), min=1e-9)
        else:
            return outputs[:, 0, :]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        entity_features: Optional[torch.Tensor] = None,
        labels_kyc: Optional[torch.Tensor] = None,
        labels_obligation: Optional[torch.Tensor] = None,
        labels_risk: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the ICC multi-head classifier.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len]
            entity_features: Entity feature tensor [batch_size, num_features]
            labels_kyc: KYC tier labels [batch_size]
            labels_obligation: Obligation labels [batch_size]
            labels_risk: Risk labels [batch_size]

        Returns:
            Dictionary with logits and optional losses.
        """
        # Transformer encoding
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        last_hidden_states = bert_output.last_hidden_state  # [batch, seq_len, hidden]

        # Pooling
        pooled = self._pool_hidden_states(last_hidden_states, attention_mask)

        # Entity feature integration
        if self.use_entity_features and entity_features is not None:
            entity_emb = self.entity_embedding(entity_features)
            pooled = torch.cat([pooled, entity_emb], dim=-1)

        # Classification heads
        kyc_logits = self.kyc_head(pooled)
        obligation_logits = self.obligation_head(pooled)
        risk_logits = self.risk_head(pooled)

        output = {
            'kyc_logits': kyc_logits,
            'obligation_logits': obligation_logits,
            'risk_logits': risk_logits,
        }

        # Compute losses if labels are provided
        losses = {}
        if labels_kyc is not None:
            losses['kyc_loss'] = F.cross_entropy(kyc_logits, labels_kyc)
        if labels_obligation is not None:
            losses['obligation_loss'] = F.cross_entropy(obligation_logits, labels_obligation)
        if labels_risk is not None:
            losses['risk_loss'] = F.cross_entropy(risk_logits, labels_risk)

        if losses:
            losses['total_loss'] = sum(losses.values())
            output.update(losses)

        return output


class ICCModelWrapper:
    """
    Wrapper class providing convenience methods for loading, saving,
    and inference with the ICC model.
    """

    def __init__(self, model: ICCMultiHeadClassifier, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_pretrained(cls, model_path: str) -> 'ICCModelWrapper':
        """Load a pre-trained ICC model from a directory."""
        model = ICCMultiHeadClassifier()
        model.load_state_dict(
            torch.load(f"{model_path}/icc_model.pt", map_location='cpu')
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return cls(model, tokenizer)

    def save(self, output_dir: str):
        """Save the model and tokenizer."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.model.state_dict(), f"{output_dir}/icc_model.pt")
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")

    def predict(
        self,
        text: str,
        entity_features: Optional[torch.Tensor] = None
    ) -> Dict[str, Dict]:
        """
        Run inference on a single text input.

        Args:
            text: Input text to classify.
            entity_features: Optional entity feature tensor.

        Returns:
            Dictionary with predictions for each task.
        """
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding='max_length',
        )

        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        if entity_features is not None:
            entity_features = entity_features.to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                entity_features=entity_features,
            )

        predictions = {
            'kyc_tier': {
                'logits': outputs['kyc_logits'].cpu().numpy().tolist(),
                'predicted_class': torch.argmax(outputs['kyc_logits'], dim=-1).item(),
            },
            'obligation': {
                'logits': outputs['obligation_logits'].cpu().numpy().tolist(),
                'predicted_class': torch.argmax(outputs['obligation_logits'], dim=-1).item(),
            },
            'risk': {
                'logits': outputs['risk_logits'].cpu().numpy().tolist(),
                'predicted_class': torch.argmax(outputs['risk_logits'], dim=-1).item(),
            },
        }

        return predictions

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Run inference on a batch of texts."""
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results
