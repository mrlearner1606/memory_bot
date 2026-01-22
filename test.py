import requests

prompt = "A futuristic cityscape at sunset with flying cars and neon lights"

url = f"https://gen.pollinations.ai/text/{prompt}"

params = {
    "model": "openai-fast",
    "key": "sk_GCcYCp80g1XMLehLUYTNQFcqkW74FDgV"  # put your full key here
}

response = requests.get(url, params=params)
print(response.text)

