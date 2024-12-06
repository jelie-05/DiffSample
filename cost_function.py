import torch

class LinearCost(torch.nn.Module):
    def __init__(self,
        theta: torch.TensorType):
        super().__init__()
    
    def forward(self):
        return self._calc_costs()

    @classmethod
    def _calc_costs(cls, theta: torch.Tensor):
        return
    
    @staticmethod
    def _cost_terms():
        return
    
    def _curvature_cost():
        return
    
    def _lateral_jerk_cost():
        return
    
    def _velocity_cost():
        return
    
    def _collision_cost():
        return
    
    def _raceline_cost():
        return
    
    def _prediction_cost():
        return