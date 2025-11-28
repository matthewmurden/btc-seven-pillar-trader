from src.models.base_model import BaseModel


class ModelBreakout(BaseModel):
    def __init__(self):
        super().__init__(target_name="breakout_target")

    @classmethod
    def load(cls, path: str) -> "ModelBreakout":
        return super(ModelBreakout, cls).load(path)  # type: ignore
