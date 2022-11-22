from dataclasses import dataclass


@dataclass
class HyperparameterArgs:
    training: str = "semi-supervised"
    epoch: int = 300
    batch_size: int = 100
    batch_size_labels: int = 100
    batch_size_unlabels: int = 100
    image_per_class: int = 200
    aug: bool = True
    translation: bool = True
    flip: bool = True
    gaussian: bool = True
    ema_consistency: float = 0.999
    lr: float = 0.001
    warm_up_steps: int = 1000

    # Consistency hyperparameters
    max_consistency_cost: float = 100.0
    ema_decay_during_rampup: float = 0.99
    ema_decay_after_rampup: float = 0.999
    consistency_trust: float = 0.0

    # Training schedule
    global_step: int = 0
    rampup_length: int = 40000
    rampdown_length: int = 25000
    training_length: int = 164000

    # Optimizer parameters
    max_learning_rate: float = 0.003
    adam_beta_1_before_rampdown: float = 0.9
    adam_beta_1_after_rampdown: float = 0.5
    adam_beta_2_during_rampup: float = 0.99
    adam_beta_2_after_rampup: float = 0.999
    adam_epsilon: float = 1e-8


if __name__ == '__main__':
    hp = HyperparameterArgs()
    # hp.epoch = 100
    # print(vars(hp))
