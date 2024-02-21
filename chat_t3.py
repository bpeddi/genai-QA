import requests
from langchain import HuggingFaceHub, PromptTemplate, LLMChain
# API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
# headers = {"Authorization": "Bearer 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"}

# def query(payload):
# 	response = requests.post(API_URL, headers=headers, json=payload)
# 	return response.json()
	
# output = query({
# 	"inputs": "The BEST part about this recipe and just like our Butter Chicken — you may already have these ingredients in your kitchen. If not, they are so easy to find in any grocery store, you won’t need to go searching high and low to find them! Authentic Chicken Tikka Masala is usually made with yogurt marinated chicken, skewered and chargrilled for incredible bbq flavours. For the sake of making this recipe much easier for us to make at home, we are using a skillet or pot to cook it all in, while still keeping those amazing flavours.",
# })
# print(output)


import requests

API_URL = "https://api-inference.huggingface.co/models/maidalun1020/bce-embedding-base_v1"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": "Today is a sunny day and I will get some ice cream.",
})
print(output)