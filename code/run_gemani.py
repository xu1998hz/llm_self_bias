# import google.generativeai as genai

# genai.configure(api_key="AIzaSyD6TPDOsho_SsIGneOHNLjAyN07JCGnwyk")
# model = genai.GenerativeModel(model_name="text-bison-001")  # gemini-pro

# prompt = """
# Translate 'I want to eat hotpot' into Chinese
# """

# completion = model.generate_content(
#     prompt, generation_config={"temperature": 1.0, "max_output_tokens": 800}
# )

# print(completion.text)

import google.generativeai as palm

palm.configure(api_key="AIzaSyD6TPDOsho_SsIGneOHNLjAyN07JCGnwyk")

prompt = """
Translate "I want to go to China" into Finish"
"""

completion = palm.generate_text(
    prompt=prompt,
    temperature=1.0,
    # The maximum length of response
    max_output_tokens=800,
)

print(completion.result)
