"""
Production Model Version Management and Deployment
===================================================

Handles safe model versioning, deployment strategies (shadow mode, A/B testing),
rollback mechanisms, and persistent model storage.

Author: Production ML System
Version: 1.0
"""

import pickle
import dill
import json
import shutil
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from pathlib import Path
from loguru import logger
import threading
from dataclasses import dataclass, asdict
import hashlib


@dataclass
class ModelVersion:
    """Metadata for a model version"""
    version_id: str
    timestamp: datetime
    model_name: str
    model_type: str
    training_samples: int
    performance_metrics: Dict[str, float]
    file_path: str
    status: str  # 'active', 'shadow', 'archived', 'failed'
    deployment_strategy: str
    parent_version: Optional[str] = None
    validation_results: Optional[Dict[str, Any]] = None


class ModelUpdater:
    """
    Manages model lifecycle: versioning, deployment, validation, and rollback.
    """
    
    def __init__(self, config: Dict[str, Any], models_dir: str = "./models"):
        self.config = config.get('model_retraining', {}).get('deployment', {})
        self.models_dir = Path(models_dir)
        self.versions_dir = self.models_dir / "versions"
        self.active_dir = self.models_dir / "active"
        
        # Create directory structure
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.active_dir.mkdir(parents=True, exist_ok=True)
        
        # Deployment configuration
        self.deployment_strategy = self.config.get('strategy', 'shadow_mode')
        self.validation_trades = self.config.get('validation_trades', 50)
        self.rollback_threshold = self.config.get('rollback_threshold', 0.05)
        
        # Model registry
        self.model_registry: Dict[str, ModelVersion] = {}
        self.active_models: Dict[str, Any] = {}
        self.shadow_models: Dict[str, Any] = {}
        
        # A/B testing state
        self.ab_test_active = False
        self.ab_test_results: Dict[str, List[float]] = {'A': [], 'B': []}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Load existing registry
        self._load_registry()
        
        logger.info(f"Model updater initialized with strategy: {self.deployment_strategy}")
    
    def save_new_model_version(self, models: List[Dict[str, Any]], 
                               training_samples: int,
                               performance_metrics: Dict[str, float]) -> List[str]:
        """
        Save newly trained models as versioned artifacts.
        
        Args:
            models: List of trained model dictionaries
            training_samples: Number of samples used for training
            performance_metrics: Performance metrics from training
            
        Returns:
            List of version IDs for saved models
        """
        with self._lock:
            version_ids = []
            
            for model_dict in models:
                try:
                    version_id = self._generate_version_id(model_dict)
                    model_name = model_dict.get('name', 'unnamed_model')
                    
                    logger.info(f"Saving model version: {model_name} (v{version_id})")
                    
                    # Create version directory
                    version_dir = self.versions_dir / version_id
                    version_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save model using multiple strategies
                    model_file = self._save_model_artifact(model_dict, version_dir, version_id)
                    
                    if not model_file:
                        logger.error(f"Failed to save model {model_name}")
                        continue
                    
                    # Create version metadata
                    version = ModelVersion(
                        version_id=version_id,
                        timestamp=datetime.now(),
                        model_name=model_name,
                        model_type=model_dict.get('strategy', 'unknown'),
                        training_samples=training_samples,
                        performance_metrics=performance_metrics,
                        file_path=str(model_file),
                        status='pending',
                        deployment_strategy=self.deployment_strategy
                    )
                    
                    # Save metadata
                    metadata_file = version_dir / "metadata.json"
                    with open(metadata_file, 'w') as f:
                        json.dump(asdict(version), f, indent=2, default=str)
                    
                    # Save model configuration
                    config_file = version_dir / "config.json"
                    with open(config_file, 'w') as f:
                        json.dump({
                            'name': model_dict.get('name'),
                            'strategy': model_dict.get('strategy'),
                            'parameters': model_dict.get('parameters', {}),
                            'features': model_dict.get('features', [])
                        }, f, indent=2, default=str)
                    
                    # Register version
                    self.model_registry[version_id] = version
                    version_ids.append(version_id)
                    
                    logger.info(f"✓ Saved model version {version_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to save model {model_dict.get('name', 'unnamed')}: {e}")
            
            # Save updated registry
            self._save_registry()
            
            return version_ids
    
    def _save_model_artifact(self, model_dict: Dict[str, Any], 
                            version_dir: Path, version_id: str) -> Optional[Path]:
        """Save model artifact using best available method"""
        model = model_dict.get('model')
        
        if not model:
            return None
        
        # Try pickle first
        try:
            model_file = version_dir / f"{version_id}.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            logger.debug(f"Saved with pickle: {model_file}")
            return model_file
        except Exception as e:
            logger.debug(f"Pickle save failed: {e}")
        
        # Try dill
        try:
            model_file = version_dir / f"{version_id}.dill"
            with open(model_file, 'wb') as f:
                dill.dump(model, f)
            logger.debug(f"Saved with dill: {model_file}")
            return model_file
        except Exception as e:
            logger.debug(f"Dill save failed: {e}")
        
        # Fallback: save serializable components
        try:
            if isinstance(model, dict):
                serializable = {}
                for key, value in model.items():
                    try:
                        pickle.dumps(value)
                        serializable[key] = value
                    except:
                        pass
                
                model_file = version_dir / f"{version_id}_components.pkl"
                with open(model_file, 'wb') as f:
                    pickle.dump(serializable, f)
                logger.warning(f"Saved partial model components: {model_file}")
                return model_file
        except Exception as e:
            logger.error(f"All save methods failed: {e}")
        
        return None
    
    def deploy_model(self, version_id: str, force: bool = False) -> bool:
        """
        Deploy a model version using configured deployment strategy.
        
        Args:
            version_id: Version ID to deploy
            force: Skip validation if True
            
        Returns:
            bool: Success status
        """
        with self._lock:
            if version_id not in self.model_registry:
                logger.error(f"Version {version_id} not found in registry")
                return False
            
            version = self.model_registry[version_id]
            
            logger.info(f"Deploying model {version.model_name} v{version_id} "
                       f"with strategy: {self.deployment_strategy}")
            
            # Load model
            model = self._load_model(version_id)
            if not model:
                logger.error(f"Failed to load model {version_id}")
                return False
            
            # Deploy based on strategy
            if self.deployment_strategy == 'immediate' or force:
                return self._deploy_immediate(version_id, model, version)
            
            elif self.deployment_strategy == 'shadow_mode':
                return self._deploy_shadow(version_id, model, version)
            
            elif self.deployment_strategy == 'a_b_test':
                return self._deploy_ab_test(version_id, model, version)
            
            else:
                logger.error(f"Unknown deployment strategy: {self.deployment_strategy}")
                return False
    
    def _deploy_immediate(self, version_id: str, model: Any, 
                         version: ModelVersion) -> bool:
        """Immediately activate new model"""
        try:
            # Archive current active model
            self._archive_active_models()
            
            # Set as active
            self.active_models[version.model_name] = model
            version.status = 'active'
            
            # Copy to active directory
            self._copy_to_active(version_id)
            
            logger.info(f"✓ Model {version_id} deployed immediately")
            self._save_registry()
            return True
            
        except Exception as e:
            logger.error(f"Immediate deployment failed: {e}")
            return False
    
    def _deploy_shadow(self, version_id: str, model: Any, 
                      version: ModelVersion) -> bool:
        """Deploy in shadow mode for validation"""
        try:
            self.shadow_models[version.model_name] = {
                'model': model,
                'version_id': version_id,
                'predictions': [],
                'trade_count': 0
            }
            
            version.status = 'shadow'
            
            logger.info(f"✓ Model {version_id} deployed in shadow mode "
                       f"({self.validation_trades} trades for validation)")
            
            self._save_registry()
            return True
            
        except Exception as e:
            logger.error(f"Shadow deployment failed: {e}")
            return False
    
    def _deploy_ab_test(self, version_id: str, model: Any, 
                       version: ModelVersion) -> bool:
        """Deploy for A/B testing"""
        try:
            self.shadow_models[f"{version.model_name}_B"] = {
                'model': model,
                'version_id': version_id,
                'predictions': [],
                'trade_count': 0
            }
            
            self.ab_test_active = True
            self.ab_test_results = {'A': [], 'B': []}
            version.status = 'ab_testing'
            
            logger.info(f"✓ Model {version_id} deployed for A/B testing")
            self._save_registry()
            return True
            
        except Exception as e:
            logger.error(f"A/B deployment failed: {e}")
            return False
    
    def validate_shadow_model(self, model_name: str) -> bool:
        """
        Validate shadow model performance and promote if successful.
        
        Returns:
            bool: True if model promoted to active
        """
        with self._lock:
            if model_name not in self.shadow_models:
                return False
            
            shadow_info = self.shadow_models[model_name]
            
            if shadow_info['trade_count'] < self.validation_trades:
                logger.debug(f"Shadow model validation: {shadow_info['trade_count']}/{self.validation_trades} trades")
                return False
            
            # Calculate shadow model performance
            predictions = shadow_info['predictions']
            correct = sum(1 for p in predictions if p.get('correct', False))
            accuracy = correct / len(predictions) if predictions else 0
            
            # Compare with active model
            active_model = self.active_models.get(model_name)
            version_id = shadow_info['version_id']
            version = self.model_registry[version_id]
            
            promote = True
            
            if active_model:
                # Get active model version
                active_version_id = self._get_active_version_id(model_name)
                if active_version_id:
                    active_version = self.model_registry[active_version_id]
                    active_accuracy = active_version.performance_metrics.get('prediction_accuracy', 0)
                    
                    # Check if new model is better
                    if accuracy < active_accuracy * (1 - self.rollback_threshold):
                        logger.warning(f"Shadow model underperforming: {accuracy:.3f} vs {active_accuracy:.3f}")
                        promote = False
            
            if promote:
                logger.info(f"✓ Promoting shadow model {version_id} to active (accuracy: {accuracy:.3f})")
                
                # Archive current active
                self._archive_active_models()
                
                # Promote shadow to active
                self.active_models[model_name] = shadow_info['model']
                version.status = 'active'
                version.validation_results = {
                    'validation_trades': shadow_info['trade_count'],
                    'validation_accuracy': accuracy,
                    'promoted_at': datetime.now().isoformat()
                }
                
                # Copy to active directory
                self._copy_to_active(version_id)
                
                # Remove from shadow
                del self.shadow_models[model_name]
                
                self._save_registry()
                return True
            else:
                logger.info(f"Shadow model {version_id} did not pass validation")
                version.status = 'failed_validation'
                del self.shadow_models[model_name]
                self._save_registry()
                return False
    
    def record_shadow_prediction(self, model_name: str, prediction: int, 
                                 actual_outcome: bool) -> None:
        """Record prediction from shadow model for validation"""
        with self._lock:
            if model_name in self.shadow_models:
                self.shadow_models[model_name]['predictions'].append({
                    'prediction': prediction,
                    'correct': actual_outcome,
                    'timestamp': datetime.now().isoformat()
                })
                self.shadow_models[model_name]['trade_count'] += 1
                
                # Check if ready for validation
                if self.shadow_models[model_name]['trade_count'] >= self.validation_trades:
                    self.validate_shadow_model(model_name)
    
    def rollback_to_version(self, version_id: str) -> bool:
        """Rollback to a previous model version"""
        with self._lock:
            logger.warning(f"Rolling back to version {version_id}")
            
            if version_id not in self.model_registry:
                logger.error(f"Version {version_id} not found")
                return False
            
            version = self.model_registry[version_id]
            model = self._load_model(version_id)
            
            if not model:
                logger.error(f"Failed to load version {version_id}")
                return False
            
            # Archive current active
            self._archive_active_models()
            
            # Restore old version
            self.active_models[version.model_name] = model
            version.status = 'active'
            
            self._copy_to_active(version_id)
            self._save_registry()
            
            logger.info(f"✓ Rolled back to version {version_id}")
            return True
    
    def get_active_model(self, model_name: str) -> Optional[Any]:
        """Get currently active model"""
        with self._lock:
            return self.active_models.get(model_name)
    
    def get_model_versions(self, model_name: Optional[str] = None) -> List[ModelVersion]:
        """Get all versions, optionally filtered by model name"""
        with self._lock:
            versions = list(self.model_registry.values())
            
            if model_name:
                versions = [v for v in versions if v.model_name == model_name]
            
            return sorted(versions, key=lambda v: v.timestamp, reverse=True)
    
    def _load_model(self, version_id: str) -> Optional[Any]:
        """Load model from disk"""
        try:
            version = self.model_registry[version_id]
            file_path = Path(version.file_path)
            
            if not file_path.exists():
                logger.error(f"Model file not found: {file_path}")
                return None
            
            # Try appropriate loader based on file extension
            if file_path.suffix == '.pkl':
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
            elif file_path.suffix == '.dill':
                with open(file_path, 'rb') as f:
                    return dill.load(f)
            else:
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
                    
        except Exception as e:
            logger.error(f"Failed to load model {version_id}: {e}")
            return None
    
    def _generate_version_id(self, model_dict: Dict[str, Any]) -> str:
        """Generate unique version ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = model_dict.get('name', 'model')
        
        # Create hash of model configuration
        config_str = json.dumps({
            'name': model_name,
            'strategy': model_dict.get('strategy'),
            'parameters': model_dict.get('parameters', {})
        }, sort_keys=True)
        
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        return f"{timestamp}_{config_hash}"
    
    def _copy_to_active(self, version_id: str) -> None:
        """Copy model version to active directory"""
        try:
            version = self.model_registry[version_id]
            src = Path(version.file_path)
            dst = self.active_dir / src.name
            
            shutil.copy2(src, dst)
            
            # Also copy metadata
            src_meta = src.parent / "metadata.json"
            dst_meta = self.active_dir / "metadata.json"
            shutil.copy2(src_meta, dst_meta)
            
        except Exception as e:
            logger.error(f"Failed to copy to active directory: {e}")
    
    def _archive_active_models(self) -> None:
        """Archive currently active models"""
        for model_name in list(self.active_models.keys()):
            version_id = self._get_active_version_id(model_name)
            if version_id:
                self.model_registry[version_id].status = 'archived'
        
        self.active_models.clear()
    
    def _get_active_version_id(self, model_name: str) -> Optional[str]:
        """Get version ID of currently active model"""
        for version_id, version in self.model_registry.items():
            if version.model_name == model_name and version.status == 'active':
                return version_id
        return None
    
    def _save_registry(self) -> None:
        """Save model registry to disk"""
        try:
            registry_file = self.models_dir / "model_registry.json"
            registry_data = {
                version_id: asdict(version) 
                for version_id, version in self.model_registry.items()
            }
            
            with open(registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    def _load_registry(self) -> None:
        """Load model registry from disk"""
        try:
            registry_file = self.models_dir / "model_registry.json"
            
            if registry_file.exists():
                with open(registry_file, 'r') as f:
                    registry_data = json.load(f)
                
                for version_id, data in registry_data.items():
                    # Convert timestamp string back to datetime
                    data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                    self.model_registry[version_id] = ModelVersion(**data)
                
                logger.info(f"Loaded {len(self.model_registry)} model versions from registry")
                
        except Exception as e:
            logger.warning(f"Could not load registry: {e}")
