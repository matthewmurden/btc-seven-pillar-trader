from src.models.base_model import BaseModel


class Model1m(BaseModel):
    def __init__(self):
        super().__init__(target_name="next_1m_up")

    @classmethod
    def load(cls, path: str) -> "Model1m":
        return super(Model1m, cls).load(path)  # type: ignore
