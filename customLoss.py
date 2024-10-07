from tensorflow.keras.losses import Loss
from tensorflow import Tensor
from typing import Tuple

class CustomLoss(Loss):
    def __init__(self, l1:Loss, l2:Loss, alpha:float):
        '''Initializes the Custom Loss

        Args:
            l1 [Loss]: The first loss function
            l2 [Loss]: The second loss function
            alpha [float]: The weight of the second loss function
        '''
        super(CustomLoss, self).__init__()
        self.l1 = l1
        self.l2 = l2
        self.alpha = alpha

    def call(self, y_true:Tuple[Tensor, Tensor], y_pred:Tuple[Tensor, Tensor], sample_weight=None) -> Tensor:
        '''Calculates the loss

        Args:
            y_true [Tuple[Tensor, Tensor]]: The true values
            y_pred [Tuple[Tensor, Tensor]]: The predicted values
            sample_weight: The weight of each sample

        Returns:
            Tensor: The loss
        '''
        return self.l1(y_true[0], y_pred[0], sample_weight) + self.alpha * self.l2(y_true[1], y_pred[1], sample_weight)
    
    def get_config(self):
        '''Gets the configuration of the loss
        '''
        return {
            'l1': self.l1,
            'l2': self.l2,
            'alpha': self.alpha
        }
    
    @classmethod
    def from_config(cls, config):
        '''Creates a CustomLoss from a configuration
        '''
        return cls(**config)
