import pixie.sdk as pixie


@pixie.app
def hello():
    """
    My first Pixie app.
    """
    return "hello"


pixie.create_prompt("my_first_prompt", description="My first prompt")
