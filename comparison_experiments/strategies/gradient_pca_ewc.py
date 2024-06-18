import torch
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin



def GradientPCAEWCPlugin(SupervisedPlugin):
    def __init__(
            self, 
            lambda_ewc=100., 
            grad_est_episodes=1,
            ):

        super().__init__()
        self.lambda_ewc = lambda_ewc
        self.grad_est_episodes = grad_est_episodes
        self.saved_parameters = []
        self.importances = []
        self.projections = []

    def compute_ewc_penalty(self, old_weights, new_weights, fisher, projection):
        # make sure projection does not get gradients ever
        projection.requires_grad = False
        w_anchor = old_weights.detach().reshape(-1)
        w = new_weights.detach().reshape(-1)   
        w_anchor_proj = w_anchor @ projection
        w_proj = w @ projection
        square_diff = (w_proj - w_anchor_proj)**2
        return (fisher * square_diff).sum()


    def before_backward(self, strategy, **kwargs):
        ewc_loss = 0

        # we make use of the parameter groups set up in our model files
        # assert here that the chosen model has this set up
        assert hasattr(strategy.model, "get_parameter_groups"), "missing get_parameter_groups function in model"
        parameter_groups = strategy.model.get_parameter_groups()[0]

        # loop over all old parameters and compute ewc loss
        for saved_param, fisher_infos, projections in zip(self.saved_parameters, self.importances, self.projections):
            for old_param, new_param, fisher_info, projection in zip(saved_param, parameter_groups, fisher_infos, projections):
                ewc_loss += self.compute_ewc_penalty(old_param, new_param, fisher_info, projection)
        
        # add ewc loss to strategy loss
        strategy.loss += self.lambda_ewc * ewc_loss
    
    def after_training_exp(self, strategy, **kwargs):
        # update saved parameters and compute new fisher information and projection matrices
        # get first parameter group, for this we want to compute ewc
        parameter_group = strategy.model.get_parameter_groups()[0]['params']
        # register a gradient hook for each parameter in the first parameter group
        for i, param in parameter_group:
            param.register_hook(lambda grad: grad.clone().detach())
        
        # iterate over the current dataset and store gradients for each parameter
        # create a dictionary with an empty list for each parameter
        gradients = {}
        for param in parameter_groups:
            gradients[param] = []
        for i in range(self.grad_est_episodes):
            x, y, t = next(strategy.experience.train_stream)
            strategy.model.zero_grad()
            x,y,t = x.to(strategy.device), y.to(strategy.device), t.to(strategy.device)
            strategy.model(x, t).backward()






        
