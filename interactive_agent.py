from types import NoneType
from pixie import pixie_app, UserInput
from pixie.types import PixieGenerator


@pixie_app
async def interactive_agent(_: NoneType) -> PixieGenerator[str, str]:
    yield "Hello! I'm your interactive agent. How can I assist you today?"
    while True:
        user_input = yield UserInput(str)
        if user_input.lower() in {"exit", "quit", "stop"}:
            yield "Goodbye!"
            break
        yield f"You said: {user_input}"
        yield "blah blah blah. What do you think about that?"
