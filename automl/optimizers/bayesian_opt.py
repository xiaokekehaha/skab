from typing import Dict, Any, Callable
import optuna
from ..base.base_optimizer import BaseOptimizer

class BayesianOptimizer(BaseOptimizer):
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        
    def optimize(self,
                objective: Callable,
                param_space: Dict[str, Any],
                n_trials: int = 100) -> Dict[str, Any]:
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=self.seed)
        )
        
        study.optimize(objective, n_trials=n_trials)
        return study.best_params 