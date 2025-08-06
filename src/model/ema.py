import torch


class EMAWrapper(object):
    """A wrapper class for exponential moving average of model weights."""

    def __init__(
        self, model: torch.nn.Module, decay: float = 0.999, mutable_param_keywords=None
    ):
        """
        model: a pytorch model to apply EMA
        decay: a scaler to indicate the decay rate
        mutable_param_keywords: keywords of parameters to apply EMA decay, other params will stay untouched
        """
        self.model = model
        self.decay = decay
        self.mutable_param_keywords = [
            s.strip() for s in mutable_param_keywords if s.strip()
        ]
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if self.mutable_param_keywords and not any(
                [keyword in name for keyword in self.mutable_param_keywords]
            ):
                continue
            assert name in self.shadow
            new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[
                name
            ]
            self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            assert name in self.shadow
            self.backup[name] = param.data
            param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            assert name in self.backup
            param.data = self.backup[name]
        self.backup = {}