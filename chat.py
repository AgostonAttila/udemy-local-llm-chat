import chainlit as cl
from typing import List


from ctransformers import AutoModelForCausalLM


def get_prompt(instruction: str, history: List[str] = None) -> str:
    system = "You are an AI assistant that follows instruction extremely well. Help as much as you can. Give short answers."
    prompt = f"### System:\n{system}\n\n### User:\n"
    if history is not None:
        prompt += f"This is the conversation history: {''.join(history)}, Now answer the question:"
    prompt += f"{instruction}\n\n### Response:\n"
    print(prompt)
    return prompt


@cl.on_message
async def on_message(message: cl.Message):
    msg = cl.Message(content="")
    await msg.send()

    prompt = get_prompt(message.content)
    for word in llm(prompt, stream=True):
        await msg.stream_token(word)
    await msg.update()


@cl.on_chat_start
async def on_chat_start():
    global llm

    llm = AutoModelForCausalLM.from_pretrained(
        "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
    )
    await cl.Message("Model initialized. How can I help you?").send()

""""
history = []

question = "Which city is the capital of India"

answer = ""
for word in llm(get_prompt(question), stream=True):
    print(word, end="", flush=True)
    answer += word
print()

history.append(answer)

question = "And which is Hungary"

for word in llm(get_prompt(question, history), stream=True):
    print(word, end="", flush=True)
print()
"""
