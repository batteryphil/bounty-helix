def read_ai_breakthroughs():
    url = "https://huggingface.co/docs/diffusion-models/hermes-3#attention"
    response = requests.get(url)
    content = response.text
    return content


def write_ai_breakthroughs(path, breakthroughs):
    with open(path, "w") as file:
        file.write(breakthroughs)


ai_breakthroughs = read_ai_breakthroughs()

write_ai_breakthroughs("ai_breakthroughs.txt", ai_breakthroughs)