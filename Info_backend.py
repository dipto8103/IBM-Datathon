from groq import Groq

class LLM:
    def __init__(self):
        self.client = Groq(
        api_key="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",
        )

    def get_info(self, query, model_name = "llama-3.1-70b-versatile"):
        sense = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": query,
                }
            ],
            model = model_name,
            temperature = 1,
            max_tokens = 1024,
            top_p = 1,
            stop = None,
        )

        return sense.choices[0].message.content