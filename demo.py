import asyncio
import dotenv
from pydantic import BaseModel
from pydantic_ai import Agent, ModelMessage
import pixie.sdk as pixie
from pixie.server_utils import setup_logging
from pixie.session import client


@pixie.app
def hello():
    """
    My first Pixie app.
    """
    return "hello"


pixie.create_prompt("my_first_prompt", description="My first prompt")


class Feedback(BaseModel):
    rating: int
    comments: str | None = None


@client.session
async def my_program():
    agent = Agent(model="gpt-4o-mini", system_prompt="You're a helpful assistant.")
    await client.print("How can I assist you today? Type exit to end the conversation.")
    messages: list[ModelMessage] = []
    while True:
        user_input = await client.input(expected_type=str)
        if user_input == "exit":
            break
        response = await agent.run(user_input, message_history=messages)
        await client.print(response.output)
        messages = response.all_messages()


if __name__ == "__main__":
    setup_logging("debug")
    dotenv.load_dotenv()
    asyncio.run(my_program())
