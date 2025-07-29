'''
https://github.com/openai/openai-python

openai's api is very simple, with 3 step:
1. create an client with OpenAI or AsyncOpenAI or AzureOpenAI
2. request with "client.responses.create" or "client.chat.completions.create", then get response
3. have fun with your response

The client is essentially a wrapper for httpx.  All requests get turned into standard HTTP requests, 
and the responses are also formatted for more user-friendly.
ok
'''

import os
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# --------------- response style ----------------
def test_chat(client):
    response = client.responses.create(
        model="gpt-4o",
        instructions="You are a coding assistant that talks like a pirate.",
        input="How do I check if a Python object is an instance of a class?",
    )

    print(response.output_text)

def test_vision(client):
    prompt = "What is in this image?"
    img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/2023_06_08_Raccoon1.jpg/1599px-2023_06_08_Raccoon1.jpg"

    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": f"{img_url}"},
                ],
            }
        ],
    )


# ---------------- chat completions api ------------------
def test_completion_chat(client):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "developer", "content": "Talk like a pirate."},
            {
                "role": "user",
                "content": "How do I check if a Python object is an instance of a class?",
            },
        ],
    )

    print(completion.choices[0].message.content)


# --------------------- async openai -----------------
def test_async_chat():
    '''
    the async api is identical with sync. only different with client.
    '''
    import os
    import asyncio
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )


    async def async_chat() -> None:
        response = await client.responses.create(
            model="gpt-4o", input="Explain disestablishmentarianism to a smart five year old."
        )
        print(response.output_text)


    asyncio.run(async_chat())

# ----------------- Azure OpenAI
def test_azure():
    from openai import AzureOpenAI

    # gets the API Key from environment variable AZURE_OPENAI_API_KEY
    client = AzureOpenAI(
        # https://learn.microsoft.com/azure/ai-services/openai/reference#rest-api-versioning
        api_version="2023-07-01-preview",
        # https://learn.microsoft.com/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal#create-a-resource
        azure_endpoint="https://example-endpoint.openai.azure.com",
    )

    completion = client.chat.completions.create(
        model="deployment-name",  # e.g. gpt-35-instant
        messages=[
            {
                "role": "user",
                "content": "How do I output all files in a directory using Python?",
            },
        ],
    )
    print(completion.to_json())