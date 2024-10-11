from openai import OpenAI
client = OpenAI(api_key='sk-I-OKo5WoYX5FnJMnZhU-JV7RkJFqZNvwBEqHORPug5T3BlbkFJ5xcABHEGB7oPodSzGTM0lEuhIejDvL_nFq-Ejgl4QA')

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Write a haiku about recursion in programming."
        }
    ]
)

print(completion.choices[0].message)