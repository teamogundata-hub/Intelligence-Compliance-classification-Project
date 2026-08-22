"""
ICC Evaluation Module
======================
Comprehensive evaluation pipeline for the ICC classifier.
Produces confusion matrices, classification reports, and
per-class metrics for all three classification heads.

Author: Team Ogun — ICC Product
Date: August 2026
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
)
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Comprehensive evaluation results for a single task."""
    task_name: str
    accuracy: float
    macro_f1: float
    weighted_f1: float
    macro_precision: float
    macro_recall: float
    weighted_precision: float
    weighted_recall: float
    cohen_kappa: float
    mcc: float
    per_class_report: dict


class ICCEvaluator:
    """
    Evaluates the ICC multi-head classifier on a test dataset.
    Produces detailed reports and visualizations for each task.
    """

    KYC_CLASS_NAMES = ['Tier 1', 'Tier 2', 'Tier 3']
    OBLIGATION_CLASS_NAMES = [
        'KYC/Due Diligence', 'Customer Identification', 'Transaction Monitoring',
        'Suspicious Activity Reporting', 'Record Retention', 'Enhanced Due Diligence',
        'PEP Screening', 'Sanctions Screening', 'Cross-Border Transactions',
        'BVN/NIN Verification', 'Beneficial Ownership', 'Annual Reporting',
        'Internal Control', 'Staff Training', 'Risk Assessment',
    ]
    RISK_CLASS_NAMES = ['Compliant', 'Suspicious']

    def __init__(
        self,
        model: nn.Module,
        test_dataset,
        tokenizer,
        batch_size: int = 16,
        output_dir: str = "./evaluation_results",
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model.eval()

    def _generate_predictions(self) -> Dict[str, Dict[str, list]]:
        """Generate predictions for all tasks."""
        all_preds = {'kyc': [], 'obligation': [], 'risk': []}
        all_labels = {'kyc': [], 'obligation': [], 'risk': []}
        all_probs = {'kyc': [], 'obligation': [], 'risk': []}

        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                # Collect predictions and probabilities
                for task in ['kyc', 'obligation', 'risk']:
                    logits = outputs[f'{task}_logits']
                    probs = torch.softmax(logits, dim=1)

                    all_preds[task].extend(torch.argmax(logits, dim=1).cpu().numpy())
                    all_labels[task].extend(batch[f'{task}_labels'].numpy())
                    all_probs[task].extend(probs.cpu().numpy())

        return all_preds, all_labels, all_probs

    def evaluate_task(
        self,
        task_name: str,
        labels: list,
        predictions: list,
        class_names: List[str],
    ) -> EvaluationResult:
        """
        Evaluate a single classification task.

        Args:
            task_name: Name of the task ('kyc', 'obligation', 'risk').
            labels: True labels.
            predictions: Predicted labels.
            class_names: List of class names for this task.

        Returns:
            EvaluationResult with all metrics.
        """
        report = classification_report(
            labels, predictions,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )

        # Remove the 'accuracy' key from per-class report
        per_class = {k: v for k, v in report.items() if k not in ('accuracy', 'macro avg', 'weighted avg')}

        result = EvaluationResult(
            task_name=task_name,
            accuracy=report.get('accuracy', 0.0),
            macro_f1=report['macro avg']['f1-score'],
            weighted_f1=report['weighted avg']['f1-score'],
            macro_precision=report['macro avg']['precision'],
            macro_recall=report['macro avg']['recall'],
            weighted_precision=report['weighted avg']['precision'],
            weighted_recall=report['weighted avg']['recall'],
            cohen_kappa=cohen_kappa_score(labels, predictions),
            mcc=matthews_corrcoef(labels, predictions),
            per_class_report=per_class,
        )

        return result

    def run_full_evaluation(self) -> Dict[str, EvaluationResult]:
        """Run evaluation on all three tasks."""
        preds, labels, probs = self._generate_predictions()

        results = {}

        # KYC Tier Evaluation
        results['kyc'] = self.evaluate_task(
            'kyc', labels['kyc'], preds['kyc'], self.KYC_CLASS_NAMES
        )

        # Obligation Mapping Evaluation
        results['obligation'] = self.evaluate_task(
            'obligation', labels['obligation'], preds['obligation'], self.OBLIGATION_CLASS_NAMES
        )

        # Risk Flagging Evaluation
        results['risk'] = self.evaluate_task(
            'risk', labels['risk'], preds['risk'], self.RISK_CLASS_NAMES
        )

        return results

    def print_classification_reports(self):
        """Print formatted classification reports to stdout."""
        preds, labels, probs = self._generate_predictions()

        print("\n" + "=" * 80)
        print("ICC SYSTEM EVALUATION RESULTS")
        print("=" * 80)

        print("\n--- KYC Tier Classification ---")
        print(classification_report(
            labels['kyc'], preds['kyc'],
            target_names=self.KYC_CLASS_NAMES,
            zero_division=0,
        ))

        print("\n--- Obligation Mapping ---")
        print(classification_report(
            labels['obligation'], preds['obligation'],
            target_names=self.OBLIGATION_CLASS_NAMES,
            zero_division=0,
        ))

        print("\n--- Risk Flagging ---")
        print(classification_report(
            labels['risk'], preds['risk'],
            target_names=self.RISK_CLASS_NAMES,
            zero_division=0,
        ))

    def plot_confusion_matrices(self, save: bool = True):
        """Generate and save confusion matrices for all tasks."""
        preds, labels, probs = self._generate_predictions()

        tasks = [
            ('kyc', labels['kyc'], preds['kyc'], self.KYC_CLASS_NAMES),
            ('obligation', labels['obligation'], preds['obligation'], self.OBLIGATION_CLASS_NAMES),
            ('risk', labels['risk'], preds['risk'], self.RISK_CLASS_NAMES),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for idx, (task_name, true_labels, pred_labels, class_names) in enumerate(tasks):
            cm = confusion_matrix(true_labels, pred_labels, labels=range(len(class_names)))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=axes[idx],
            )
            axes[idx].set_title(f'{task_name.upper()} Confusion Matrix')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')

        plt.tight_layout()

        if save:
            filepath = self.output_dir / "confusion_matrices.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Confusion matrices saved to {filepath}")

        plt.close()

    def plot_per_class_f1_scores(self, save: bool = True):
        """Plot per-class F1 scores for each task."""
        preds, labels, probs = self._generate_predictions()

        tasks = [
            ('kyc', labels['kyc'], preds['kyc'], self.KYC_CLASS_NAMES),
            ('risk', labels['risk'], preds['risk'], self.RISK_CLASS_NAMES),
        ]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for idx, (task_name, true_labels, pred_labels, class_names) in enumerate(tasks):
            # Per-class F1
            f1_per_class = {}
            for i, class_name in enumerate(class_names):
                class_mask = [l == i for l in true_labels]
                if any(class_mask):
                    f1_per_class[class_name] = f1_score(
                        [l == i for l in true_labels],
                        [p == i for p in pred_labels],
                        zero_division=0,
                    )
                else:
                    f1_per_class[class_name] = 0.0

            axes[idx].bar(f1_per_class.keys(), f1_per_class.values(), color='steelblue')
            axes[idx].set_title(f'{task_name.upper()} Per-Class F1 Score')
            axes[idx].set_ylim(0, 1.0)
            axes[idx].set_ylabel('F1 Score')
            axes[idx].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save:
            filepath = self.output_dir / "per_class_f1.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Per-class F1 plots saved to {filepath}")

        plt.close()

    def save_evaluation_report(self, results: Dict[str, EvaluationResult]):
        """Save evaluation results to JSON."""
        report = {
            'evaluation_timestamp': str(__import__('datetime').datetime.utcnow()),
            'results': {
                task: asdict(result) for task, result in results.items()
            },
        }

        filepath = self.output_dir / "evaluation_report.json"
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Evaluation report saved to {filepath}")

    def benchmark_comparison(
        self,
        baseline_results: Dict[str, Dict[str, float]],
        icc_results: Dict[str, EvaluationResult],
    ) -> dict:
        """
        Compare ICC results against baseline models.

        Args:
            baseline_results: Dictionary of baseline model results.
            icc_results: Current ICC evaluation results.

        Returns:
            Comparison dictionary with improvement metrics.
        """
        comparison = {}
        for task in ['kyc', 'obligation', 'risk']:
            if task in baseline_results:
                baseline_f1 = baseline_results[task].get('weighted_f1', 0.0)
                icc_f1 = icc_results[task].weighted_f1
                improvement = icc_f1 - baseline_f1
                comparison[task] = {
                    'baseline_f1': baseline_f1,
                    'icc_f1': icc_f1,
                    'improvement': improvement,
                    'improvement_pct': (improvement / max(baseline_f1, 0.001)) * 100,
                }

        return comparison


# =============================================================================
# Benchmark Models for Comparison
# =============================================================================

class BaselineBenchmark:
    """
    Implements baseline models for benchmarking the ICC system
    against generic and non-localized approaches.
    """

    @staticmethod
    def tfidf_baseline(texts: list, labels: list) -> Dict[str, float]:
        """
        Train a TF-IDF + Logistic Regression baseline.

        Args:
            texts: List of input texts.
            labels: List of integer labels.

        Returns:
            Dictionary with baseline metrics.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )

        vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        classifier = LogisticRegression(max_iter=1000, class_weight='balanced')
        classifier.fit(X_train_tfidf, y_train)

        predictions = classifier.predict(X_test_tfidf)

        return {
            'accuracy': accuracy_score(y_test, predictions),
            'weighted_f1': f1_score(y_test, predictions, average='weighted'),
            'macro_f1': f1_score(y_test, predictions, average='macro'),
        }

    @staticmethod
    def rule_based_baseline(texts: list, labels: list) -> Dict[str, float]:
        """
        Simple rule-based baseline using keyword matching.
        """
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )

        # Majority class baseline
        from collections import Counter
        most_common = Counter(y_train).most_common(1)[0][0]
        predictions = [most_common] * len(y_test)

        return {
            'accuracy': accuracy_score(y_test, predictions),
            'weighted_f1': f1_score(y_test, predictions, average='weighted', zero_division=0),
            'macro_f1': f1_score(y_test, predictions, average='macro', zero_division=0),
        }
