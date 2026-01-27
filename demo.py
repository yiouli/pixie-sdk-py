import asyncio
from pydantic import BaseModel
import pixie.sdk as pixie
from pixie.server import setup_logging
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
    setup_logging("debug")
    await client.print("Starting my program...")
    stopped = False
    while not stopped:
        name = await client.input("Please enter your name:")
        feedback = await client.input(
            "Please provide your feedback:",
            expected_type=Feedback,
        )
        await client.print(
            f"Feedback received: Rating={feedback.rating}, Comments={feedback.comments}"
        )
        await client.print(f"Thank you for your input, {name}!")
        stopped = feedback.comments == "stop"
    await client.print("Program ended.")


if __name__ == "__main__":
    asyncio.run(my_program())
