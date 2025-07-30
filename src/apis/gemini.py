import os

from google import genai


def test_gen_text():
    # The client gets the API key from the environment variable `GEMINI_API_KEY`.
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    response = client.models.generate_content(
        model="gemini-2.5-flash", contents="Explain how AI works in a few words"
    )
    print(response.text)


def test_content_config():
    from google.genai import types
    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="How does AI work?",
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0), # Disables thinking
            system_instruction="You are a cat. Your name is Neko.",
            temperature=0.1,
        ),
    )
    print(response.text)


def test_multimodal():
    from PIL import Image

    client = genai.Client()

    image = Image.open("/path/to/organ.png")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[image, "Tell me about this instrument"]
    )
    print(response.text)


def test_steaming():
    """
    By default, the model returns a response only after the entire generation process is complete.
    For more fluid interactions, use streaming to receive GenerateContentResponse instances incrementally as they're generated.
    """
    client = genai.Client()
    response = client.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents=["Explain how AI works"]
    )
    for chunk in response:
        print(chunk.text, end="")


def test_multi_turn_chat():
    client = genai.Client()
    chat = client.chats.create(model="gemini-2.5-flash")

    response = chat.send_message("I have 2 dogs in my house.")
    print(response.text)

    response = chat.send_message("How many paws are in my house?")
    print(response.text)

    for message in chat.get_history():
        print(f'role - {message.role}',end=": ")
        print(message.parts[0].text)