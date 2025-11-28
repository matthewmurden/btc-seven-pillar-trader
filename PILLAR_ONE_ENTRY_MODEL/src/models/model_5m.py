from src.models.base_model import BaseModel


class Model5m(BaseModel):
    def __init__(self):
        super().__init__(target_name="next_5m_up")

    @classmethod
    def load(cls, path: str) -> "Model5m":
        return super(Model5m, cls).load(path)  # type: ignore
