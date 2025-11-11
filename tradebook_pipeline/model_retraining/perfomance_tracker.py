"""
Real-Time Performance Tracking for Live Trading Models
=======================================================

Tracks prediction accuracy, execution quality, and model drift metrics
during live trading sessions with comprehensive logging and alerting.

Author: Production ML System
Version: 1.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from loguru import logger
import threading
from scipy import stats


@dataclass
class TradeOutcome:
    """Records the outcome of a trade for performance evaluation"""
    trade_id: str
    timestamp: datetime
    prediction: int
    predicted_direction: str  # 'BUY', 'SELL', 'HOLD'
    confidence: Optional[float]
    entry_price: float
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    actual_return: Optional[float] = None
    prediction_correct: Optional[bool] = None
    hold_duration_seconds: Optional[float] = None
    execution_latency_ms: Optional[float] = None
    slippage_pct: Optional[float] = None


@dataclass
class PerformanceSnapshot:
    """Snapshot of current model performance"""
    timestamp: datetime
    lookback_trades: int
    prediction_accuracy: float
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    false_negative_rate: float
    avg_return_correct: float
    avg_return_incorrect: float
    win_rate: float
    avg_execution_latency_ms: float
    avg_slippage_pct: float
    model_confidence_calibration: float
    total_trades: int
    profitable_trades: int
    unprofitable_trades: int


class PerformanceTracker:
    """
    Tracks real-time model performance during live trading with 
    comprehensive metrics and drift detection.
    """
    
    def __init__(self, config: Dict[str, Any], output_dir: str = "./monitoring"):
        self.config = config.get('model_retraining', {}).get('monitoring', {})
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Trade tracking
        self.open_trades: Dict[str, TradeOutcome] = {}
        self.completed_trades: deque = deque(maxlen=10000)
        self.recent_predictions: deque = deque(maxlen=1000)
        
        # Performance metrics
        self.lookback_window = self.config.get('lookback_trades', 100)
        self.performance_history: deque = deque(maxlen=1000)
        
        # Distribution tracking for drift detection
        self.feature_distributions: Dict[str, deque] = {}
        self.prediction_distributions: deque = deque(maxlen=1000)
        
        # Alert thresholds
        self.alert_thresholds = self.config.get('alert_thresholds', {
            'accuracy_drop': 0.10,
            'latency_increase': 2.0,
            'slippage_threshold': 0.02
        })
        
        # Baseline metrics (set after initial burn-in period)
        self.baseline_metrics: Optional[PerformanceSnapshot] = None
        self.baseline_set = False
        self.burn_in_trades = 50
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("Performance tracker initialized")
    
    def record_prediction(self, trade_id: str, prediction: int, 
                         predicted_direction: str, confidence: Optional[float],
                         entry_price: float, execution_latency_ms: Optional[float],
                         features: Optional[pd.DataFrame] = None) -> None:
        """
        Record a new prediction when a trade is opened.
        
        Args:
            trade_id: Unique identifier for the trade
            prediction: Raw model prediction output
            predicted_direction: 'BUY', 'SELL', or 'HOLD'
            confidence: Model confidence score (0-1)
            entry_price: Entry price for the trade
            execution_latency_ms: Time taken to execute trade
            features: Feature values used for prediction (for drift detection)
        """
        with self._lock:
            outcome = TradeOutcome(
                trade_id=trade_id,
                timestamp=datetime.now(),
                prediction=prediction,
                predicted_direction=predicted_direction,
                confidence=confidence,
                entry_price=entry_price,
                execution_latency_ms=execution_latency_ms
            )
            
            self.open_trades[trade_id] = outcome
            self.recent_predictions.append({
                'timestamp': datetime.now(),
                'prediction': prediction,
                'confidence': confidence,
                'direction': predicted_direction
            })
            
            # Track feature distribution for drift detection
            if features is not None:
                self._update_feature_distributions(features)
            
            logger.debug(f"Recorded prediction: {trade_id} - {predicted_direction} @ {entry_price}")
    
    def record_trade_close(self, trade_id: str, exit_price: float, 
                          slippage_pct: Optional[float] = None) -> None:
        """
        Record when a trade is closed and calculate outcome.
        
        Args:
            trade_id: Unique identifier for the trade
            exit_price: Exit price for the trade
            slippage_pct: Execution slippage percentage
        """
        with self._lock:
            if trade_id not in self.open_trades:
                logger.warning(f"Trade {trade_id} not found in open trades")
                return
            
            outcome = self.open_trades[trade_id]
            outcome.exit_price = exit_price
            outcome.exit_timestamp = datetime.now()
            outcome.slippage_pct = slippage_pct
            
            # Calculate return
            if outcome.predicted_direction == 'BUY':
                outcome.actual_return = (exit_price - outcome.entry_price) / outcome.entry_price
            elif outcome.predicted_direction == 'SELL':
                outcome.actual_return = (outcome.entry_price - exit_price) / outcome.entry_price
            else:
                outcome.actual_return = 0.0
            
            # Determine if prediction was correct
            outcome.prediction_correct = outcome.actual_return > 0
            
            # Calculate hold duration
            outcome.hold_duration_seconds = (
                outcome.exit_timestamp - outcome.timestamp
            ).total_seconds()
            
            # Move to completed trades
            self.completed_trades.append(outcome)
            del self.open_trades[trade_id]
            
            # Update performance metrics
            self._update_performance_metrics()
            
            # Check for alerts
            self._check_performance_alerts()
            
            logger.info(f"Closed trade {trade_id}: Return={outcome.actual_return:+.4f}, "
                       f"Correct={outcome.prediction_correct}")
    
    def _update_feature_distributions(self, features: pd.DataFrame) -> None:
        """Track feature distributions for drift detection"""
        for col in features.columns:
            if col not in self.feature_distributions:
                self.feature_distributions[col] = deque(maxlen=1000)
            
            try:
                value = float(features[col].iloc[0])
                if not np.isnan(value):
                    self.feature_distributions[col].append(value)
            except (ValueError, TypeError, IndexError):
                continue
    
    def _update_performance_metrics(self) -> None:
        """Calculate current performance metrics"""
        recent = list(self.completed_trades)[-self.lookback_window:]
        
        if len(recent) < 10:  # Need minimum trades for meaningful metrics
            return
        
        # Basic accuracy metrics
        total = len(recent)
        correct = sum(1 for t in recent if t.prediction_correct)
        incorrect = total - correct
        
        accuracy = correct / total if total > 0 else 0.0
        
        # Confusion matrix components
        true_positives = sum(1 for t in recent 
                           if t.prediction_correct and t.predicted_direction in ['BUY', 'SELL'])
        false_positives = sum(1 for t in recent 
                            if not t.prediction_correct and t.predicted_direction in ['BUY', 'SELL'])
        true_negatives = sum(1 for t in recent 
                           if t.prediction_correct and t.predicted_direction == 'HOLD')
        false_negatives = sum(1 for t in recent 
                            if not t.prediction_correct and t.predicted_direction == 'HOLD')
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        fpr = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0.0
        fnr = false_negatives / (false_negatives + true_positives) if (false_negatives + true_positives) > 0 else 0.0
        
        # Return metrics
        profitable = [t for t in recent if t.actual_return and t.actual_return > 0]
        unprofitable = [t for t in recent if t.actual_return and t.actual_return <= 0]
        
        win_rate = len(profitable) / total if total > 0 else 0.0
        
        avg_return_correct = np.mean([t.actual_return for t in recent if t.prediction_correct and t.actual_return]) if any(t.prediction_correct for t in recent) else 0.0
        avg_return_incorrect = np.mean([t.actual_return for t in recent if not t.prediction_correct and t.actual_return]) if any(not t.prediction_correct for t in recent) else 0.0
        
        # Execution quality
        avg_latency = np.mean([t.execution_latency_ms for t in recent if t.execution_latency_ms is not None])
        avg_slippage = np.mean([t.slippage_pct for t in recent if t.slippage_pct is not None])
        
        # Confidence calibration
        confidence_calibration = self._calculate_confidence_calibration(recent)
        
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            lookback_trades=total,
            prediction_accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            false_positive_rate=fpr,
            false_negative_rate=fnr,
            avg_return_correct=avg_return_correct,
            avg_return_incorrect=avg_return_incorrect,
            win_rate=win_rate,
            avg_execution_latency_ms=avg_latency,
            avg_slippage_pct=avg_slippage,
            model_confidence_calibration=confidence_calibration,
            total_trades=total,
            profitable_trades=len(profitable),
            unprofitable_trades=len(unprofitable)
        )
        
        self.performance_history.append(snapshot)
        
        # Set baseline after burn-in
        if not self.baseline_set and len(self.completed_trades) >= self.burn_in_trades:
            self.baseline_metrics = snapshot
            self.baseline_set = True
            logger.info(f"Baseline metrics set: Accuracy={accuracy:.3f}, Win Rate={win_rate:.3f}")
        
        # Save snapshot
        self._save_performance_snapshot(snapshot)
    
    def _calculate_confidence_calibration(self, recent_trades: List[TradeOutcome]) -> float:
        """
        Calculate how well model confidence correlates with actual accuracy.
        Returns 0-1 score where 1 = perfect calibration.
        """
        trades_with_confidence = [t for t in recent_trades if t.confidence is not None]
        
        if len(trades_with_confidence) < 10:
            return 0.5
        
        # Bin by confidence
        bins = np.linspace(0, 1, 11)
        bin_accuracies = []
        bin_confidences = []
        
        for i in range(len(bins) - 1):
            bin_trades = [t for t in trades_with_confidence 
                         if bins[i] <= t.confidence < bins[i+1]]
            
            if len(bin_trades) >= 3:
                accuracy = sum(1 for t in bin_trades if t.prediction_correct) / len(bin_trades)
                avg_conf = np.mean([t.confidence for t in bin_trades])
                bin_accuracies.append(accuracy)
                bin_confidences.append(avg_conf)
        
        if len(bin_accuracies) < 2:
            return 0.5
        
        # Calculate calibration error (lower is better)
        calibration_error = np.mean([abs(acc - conf) for acc, conf in zip(bin_accuracies, bin_confidences)])
        
        # Convert to 0-1 score (1 = perfect calibration)
        return max(0, 1 - calibration_error)
    
    def _check_performance_alerts(self) -> None:
        """Check if performance has degraded beyond alert thresholds"""
        if not self.baseline_set or len(self.performance_history) < 2:
            return
        
        current = self.performance_history[-1]
        baseline = self.baseline_metrics
        
        # Accuracy degradation
        accuracy_drop = baseline.prediction_accuracy - current.prediction_accuracy
        if accuracy_drop > self.alert_thresholds['accuracy_drop']:
            logger.warning(f"ALERT: Accuracy dropped {accuracy_drop:.3f} "
                         f"(from {baseline.prediction_accuracy:.3f} to {current.prediction_accuracy:.3f})")
        
        # Latency increase
        if baseline.avg_execution_latency_ms > 0:
            latency_ratio = current.avg_execution_latency_ms / baseline.avg_execution_latency_ms
            if latency_ratio > self.alert_thresholds['latency_increase']:
                logger.warning(f"ALERT: Execution latency increased {latency_ratio:.2f}x "
                             f"(from {baseline.avg_execution_latency_ms:.1f}ms to {current.avg_execution_latency_ms:.1f}ms)")
        
        # Slippage increase
        if current.avg_slippage_pct > self.alert_thresholds['slippage_threshold']:
            logger.warning(f"ALERT: High slippage detected: {current.avg_slippage_pct:.4f}")
    
    def detect_distribution_drift(self) -> Dict[str, float]:
        """
        Detect if feature distributions have shifted significantly using KS test.
        Returns dict of p-values for each feature (low p-value = significant drift).
        """
        drift_results = {}
        
        for feature, values in self.feature_distributions.items():
            if len(values) < 100:
                continue
            
            # Split into baseline (first 50%) and recent (last 50%)
            n = len(values)
            baseline = list(values)[:n//2]
            recent = list(values)[n//2:]
            
            # Kolmogorov-Smirnov test
            try:
                statistic, p_value = stats.ks_2samp(baseline, recent)
                drift_results[feature] = {
                    'p_value': p_value,
                    'statistic': statistic,
                    'drift_detected': p_value < 0.05
                }
            except Exception as e:
                logger.debug(f"Drift detection failed for {feature}: {e}")
        
        return drift_results
    
    def get_current_metrics(self) -> Optional[PerformanceSnapshot]:
        """Get most recent performance snapshot"""
        with self._lock:
            return self.performance_history[-1] if self.performance_history else None
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary for decision making"""
        with self._lock:
            current = self.get_current_metrics()
            
            if not current:
                return {'status': 'insufficient_data', 'completed_trades': len(self.completed_trades)}
            
            # Calculate trends
            if len(self.performance_history) >= 10:
                recent_10 = list(self.performance_history)[-10:]
                accuracy_trend = np.polyfit(range(10), [m.prediction_accuracy for m in recent_10], 1)[0]
                win_rate_trend = np.polyfit(range(10), [m.win_rate for m in recent_10], 1)[0]
            else:
                accuracy_trend = 0.0
                win_rate_trend = 0.0
            
            # Drift detection
            drift_results = self.detect_distribution_drift()
            features_with_drift = [f for f, r in drift_results.items() if r['drift_detected']]
            
            return {
                'status': 'active',
                'current_metrics': asdict(current),
                'baseline_metrics': asdict(self.baseline_metrics) if self.baseline_metrics else None,
                'trends': {
                    'accuracy_trend': accuracy_trend,
                    'win_rate_trend': win_rate_trend
                },
                'drift_detection': {
                    'features_with_drift': features_with_drift,
                    'drift_details': drift_results
                },
                'data_quality': {
                    'total_completed_trades': len(self.completed_trades),
                    'open_trades': len(self.open_trades),
                    'recent_predictions': len(self.recent_predictions)
                },
                'timestamp': datetime.now().isoformat()
            }
    
    def _save_performance_snapshot(self, snapshot: PerformanceSnapshot) -> None:
        """Save performance snapshot to file"""
        try:
            timestamp_str = snapshot.timestamp.strftime('%Y%m%d_%H%M%S')
            filepath = self.output_dir / f"performance_snapshot_{timestamp_str}.json"
            
            with open(filepath, 'w') as f:
                json.dump(asdict(snapshot), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save performance snapshot: {e}")
    
    def export_trade_outcomes(self, filepath: Optional[Path] = None) -> Path:
        """Export all completed trades to CSV"""
        if filepath is None:
            filepath = self.output_dir / f"trade_outcomes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        with self._lock:
            df = pd.DataFrame([asdict(t) for t in self.completed_trades])
            df.to_csv(filepath, index=False)
        
        logger.info(f"Exported {len(df)} trade outcomes to {filepath}")
        return filepath
    
    def get_retraining_recommendation(self) -> Dict[str, Any]:
        """
        Analyze current performance and provide retraining recommendation.
        
        Returns:
            Dict with 'should_retrain' bool and reasons
        """
        summary = self.get_metrics_summary()
        
        if summary['status'] == 'insufficient_data':
            return {
                'should_retrain': False,
                'reason': 'Insufficient data for evaluation',
                'trades_needed': self.burn_in_trades - len(self.completed_trades)
            }
        
        current = summary['current_metrics']
        baseline = summary['baseline_metrics']
        
        reasons = []
        should_retrain = False
        
        # Check accuracy degradation
        if baseline and current['prediction_accuracy'] < baseline['prediction_accuracy'] * 0.85:
            reasons.append(f"Accuracy dropped {(1 - current['prediction_accuracy']/baseline['prediction_accuracy'])*100:.1f}%")
            should_retrain = True
        
        # Check negative trends
        if summary['trends']['accuracy_trend'] < -0.01:
            reasons.append("Declining accuracy trend detected")
            should_retrain = True
        
        # Check distribution drift
        if len(summary['drift_detection']['features_with_drift']) >= 3:
            reasons.append(f"Distribution drift in {len(summary['drift_detection']['features_with_drift'])} features")
            should_retrain = True
        
        # Check poor calibration
        if current['model_confidence_calibration'] < 0.6:
            reasons.append("Poor confidence calibration")
            should_retrain = True
        
        return {
            'should_retrain': should_retrain,
            'reasons': reasons,
            'current_accuracy': current['prediction_accuracy'],
            'baseline_accuracy': baseline['prediction_accuracy'] if baseline else None,
            'trades_analyzed': current['total_trades'],
            'recommendation_timestamp': datetime.now().isoformat()
        }
