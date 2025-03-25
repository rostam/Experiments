import ollama

response = ollama.chat(
    model='llama2',
    messages=[
        {'role': 'user', 'content': 'What is the capital of France?'}
    ]
)

print(response['message']['content'])