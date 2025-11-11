"""
Intelligent Retraining Manager for Adaptive ML Models
======================================================

Manages automated model retraining decisions based on performance
degradation, data accumulation, and scheduled intervals.

Author: Production ML System
Version: 1.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
import threading
import time
import json


class RetrainingManager:
    """
    Manages retraining decisions and orchestration for continuous learning.
    """
    
    def __init__(self, config: Dict[str, Any], performance_tracker: Any, 
                 tools_manager: Any, data_processor: Any):
        self.config = config.get('model_retraining', {})
        self.triggers_config = self.config.get('triggers', {})
        
        # Dependencies
        self.performance_tracker = performance_tracker
        self.tools_manager = tools_manager
        self.data_processor = data_processor
        
        # State tracking
        self.last_retrain_time: Optional[datetime] = None
        self.retrain_count = 0
        self.training_data_buffer: List[pd.DataFrame] = []
        self.min_training_samples = self.triggers_config.get('data_accumulation', {}).get('min_new_samples', 500)
        
        # Retraining triggers
        self.performance_trigger_enabled = self.triggers_config.get('performance_degradation', {}).get('enabled', True)
        self.time_trigger_enabled = self.triggers_config.get('time_based', {}).get('enabled', True)
        self.data_trigger_enabled = self.triggers_config.get('data_accumulation', {}).get('enabled', True)
        
        # Performance thresholds
        self.accuracy_threshold = self.triggers_config.get('performance_degradation', {}).get('accuracy_threshold', 0.70)
        self.lookback_trades = self.triggers_config.get('performance_degradation', {}).get('lookback_trades', 100)
        
        # Time-based trigger
        self.retrain_interval_hours = self.triggers_config.get('time_based', {}).get('interval_hours', 24)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # History
        self.retraining_history: List[Dict[str, Any]] = []
        
        logger.info(f"Retraining manager initialized with triggers: "
                   f"Performance={self.performance_trigger_enabled}, "
                   f"Time={self.time_trigger_enabled}, "
                   f"Data={self.data_trigger_enabled}")
    
    def should_retrain(self) -> tuple[bool, List[str]]:
        """
        Evaluate all retraining triggers and decide if retraining should occur.
        
        Returns:
            Tuple of (should_retrain: bool, reasons: List[str])
        """
        with self._lock:
            reasons = []
            should_retrain = False
            
            # Trigger 1: Performance Degradation
            if self.performance_trigger_enabled:
                perf_check, perf_reasons = self._check_performance_trigger()
                if perf_check:
                    should_retrain = True
                    reasons.extend(perf_reasons)
            
            # Trigger 2: Time-Based
            if self.time_trigger_enabled:
                time_check, time_reason = self._check_time_trigger()
                if time_check:
                    should_retrain = True
                    reasons.append(time_reason)
            
            # Trigger 3: Data Accumulation
            if self.data_trigger_enabled:
                data_check, data_reason = self._check_data_trigger()
                if data_check:
                    should_retrain = True
                    reasons.append(data_reason)
            
            if should_retrain:
                logger.info(f"Retraining triggered: {', '.join(reasons)}")
            
            return should_retrain, reasons
    
    def _check_performance_trigger(self) -> tuple[bool, List[str]]:
        """Check if model performance has degraded"""
        recommendation = self.performance_tracker.get_retraining_recommendation()
        
        if recommendation['should_retrain']:
            return True, recommendation['reasons']
        
        return False, []
    
    def _check_time_trigger(self) -> tuple[bool, str]:
        """Check if enough time has elapsed since last retrain"""
        if self.last_retrain_time is None:
            return False, ""
        
        elapsed = datetime.now() - self.last_retrain_time
        elapsed_hours = elapsed.total_seconds() / 3600
        
        if elapsed_hours >= self.retrain_interval_hours:
            return True, f"Scheduled retrain ({elapsed_hours:.1f}h since last retrain)"
        
        return False, ""
    
    def _check_data_trigger(self) -> tuple[bool, str]:
        """Check if enough new training data has accumulated"""
        total_samples = sum(len(df) for df in self.training_data_buffer)
        
        if total_samples >= self.min_training_samples:
            return True, f"Accumulated {total_samples} new samples (threshold: {self.min_training_samples})"
        
        return False, ""
    
    def accumulate_training_data(self, new_data: pd.DataFrame) -> None:
        """
        Add new trading data to the training buffer.
        
        Args:
            new_data: DataFrame with recent trading data and outcomes
        """
        with self._lock:
            if new_data is not None and not new_data.empty:
                self.training_data_buffer.append(new_data.copy())
                total = sum(len(df) for df in self.training_data_buffer)
                logger.debug(f"Training buffer updated: {total} total samples")
    
    def retrain_models(self, additional_data: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
        """
        Execute model retraining with accumulated data.
        
        Args:
            additional_data: Optional additional data to include in retraining
            
        Returns:
            List of newly trained models
        """
        with self._lock:
            retrain_start = time.time()
            logger.info("="*60)
            logger.info("STARTING MODEL RETRAINING")
            logger.info("="*60)
            
            try:
                # Combine all training data
                training_data = self._prepare_training_data(additional_data)
                
                if training_data is None or len(training_data) < 100:
                    logger.warning("Insufficient training data for retraining")
                    return []
                
                logger.info(f"Retraining with {len(training_data)} samples")
                logger.info(f"Features: {list(training_data.columns)}")
                
                # Process data
                logger.info("Processing training data...")
                processed_data = self.data_processor.process_data(training_data)
                
                if processed_data is None or processed_data.empty:
                    logger.error("Data processing failed during retraining")
                    return []
                
                # Train new models
                logger.info("Training new models...")
                new_models = self.tools_manager.detect_peaks(processed_data)
                
                if not new_models:
                    logger.error("No models were trained during retraining")
                    return []
                
                logger.info(f"Successfully trained {len(new_models)} new models")
                
                # Update state
                self.last_retrain_time = datetime.now()
                self.retrain_count += 1
                
                # Clear training buffer
                self.training_data_buffer.clear()
                
                # Record retraining event
                retrain_duration = time.time() - retrain_start
                self._record_retraining_event(
                    models=new_models,
                    training_samples=len(training_data),
                    duration=retrain_duration
                )
                
                logger.info("="*60)
                logger.info(f"RETRAINING COMPLETED IN {retrain_duration:.1f}s")
                logger.info("="*60)
                
                return new_models
                
            except Exception as e:
                logger.error(f"Retraining failed: {e}", exc_info=True)
                return []
    
    def _prepare_training_data(self, additional_data: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Prepare combined training dataset"""
        datasets = []
        
        # Add buffered data
        if self.training_data_buffer:
            datasets.extend(self.training_data_buffer)
        
        # Add additional data
        if additional_data is not None and not additional_data.empty:
            datasets.append(additional_data)
        
        if not datasets:
            return None
        
        # Combine and deduplicate
        combined = pd.concat(datasets, ignore_index=True)
        
        # Remove duplicates if timestamp column exists
        time_cols = [col for col in combined.columns if 'time' in col.lower() or 'date' in col.lower()]
        if time_cols:
            combined = combined.drop_duplicates(subset=time_cols[0], keep='last')
        
        logger.info(f"Prepared {len(combined)} samples from {len(datasets)} datasets")
        return combined
    
    def _record_retraining_event(self, models: List[Dict[str, Any]], 
                                training_samples: int, duration: float) -> None:
        """Record retraining event in history"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'retrain_number': self.retrain_count,
            'training_samples': training_samples,
            'duration_seconds': duration,
            'models_trained': len(models),
            'model_names': [m.get('name', 'Unnamed') for m in models],
            'performance_before': self.performance_tracker.get_current_metrics(),
            'triggers': [reason for _, reasons in [self.should_retrain()] for reason in reasons]
        }
        
        self.retraining_history.append(event)
        
        # Save to file
        try:
            filepath = Path(self.performance_tracker.output_dir) / "retraining_history.json"
            with open(filepath, 'w') as f:
                json.dump(self.retraining_history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save retraining history: {e}")
    
    def get_retraining_stats(self) -> Dict[str, Any]:
        """Get statistics about retraining history"""
        with self._lock:
            if not self.retraining_history:
                return {
                    'total_retrains': 0,
                    'status': 'no_retraining_yet'
                }
            
            durations = [e['duration_seconds'] for e in self.retraining_history]
            samples = [e['training_samples'] for e in self.retraining_history]
            
            return {
                'total_retrains': self.retrain_count,
                'last_retrain': self.last_retrain_time.isoformat() if self.last_retrain_time else None,
                'avg_duration_seconds': np.mean(durations),
                'avg_training_samples': np.mean(samples),
                'buffered_samples': sum(len(df) for df in self.training_data_buffer),
                'time_since_last_retrain_hours': (datetime.now() - self.last_retrain_time).total_seconds() / 3600 if self.last_retrain_time else None,
                'next_scheduled_retrain_hours': self.retrain_interval_hours - ((datetime.now() - self.last_retrain_time).total_seconds() / 3600) if self.last_retrain_time else self.retrain_interval_hours
            }
    
    def force_retrain(self, reason: str = "Manual trigger") -> List[Dict[str, Any]]:
        """Manually force a retraining cycle"""
        logger.info(f"Forcing retraining: {reason}")
        return self.retrain_models()
    
    def reset_triggers(self) -> None:
        """Reset all trigger states"""
        with self._lock:
            self.last_retrain_time = datetime.now()
            self.training_data_buffer.clear()
            logger.info("Retraining triggers reset")
