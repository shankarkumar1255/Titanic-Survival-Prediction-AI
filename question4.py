from llm_feature import call_llm

system_prompt = """
You are an AI assistant that explains machine learning predictions.
Always return ONLY valid JSON.

Required fields:
{
  "prediction_label": "",
  "confidence_level": "",
  "top_reason": "",
  "second_reason": "",
  "next_step": ""
}
"""

user_prompt = """
Prediction: Survived
Probability: 0.95

Features:
Pclass = 1
Age = 25
Sex = Female
Fare = 100
"""

result = call_llm(system_prompt, user_prompt)

print(result)