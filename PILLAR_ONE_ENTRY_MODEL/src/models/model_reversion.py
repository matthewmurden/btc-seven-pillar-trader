from src.models.base_model import BaseModel


class ModelReversion(BaseModel):
    def __init__(self):
        super().__init__(target_name="reversion_target")

    @classmethod
    def load(cls, path: str) -> "ModelReversion":
        return super(ModelReversion, cls).load(path)  # type: ignore
