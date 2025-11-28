from src.models.base_model import BaseModel


class Model15m(BaseModel):
    def __init__(self):
        super().__init__(target_name="next_15m_trend")

    @classmethod
    def load(cls, path: str) -> "Model15m":
        return super(Model15m, cls).load(path)  # type: ignore
