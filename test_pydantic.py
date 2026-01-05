from groq import BaseModel
from pydantic import PrivateAttr


class Test(BaseModel):
    a: int
    _t: type | None = PrivateAttr(default=None)

    def set_t(self, value: type) -> None:
        self._t = value

    @property
    def t(self) -> type | None:
        return self._t


t = Test(a=5)
t.set_t(str)

print(t.t)
