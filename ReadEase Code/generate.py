import torch
import cohere
import config
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler


def generate():
    co = cohere.Client(config.api_key)
    model_id = "CompVis/stable-diffusion-v1-4"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, revision="fp16", scheduler=scheduler, torch_dtype=torch.float32)
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")

    words = []
    sentences = []
    i = 0
    prompt = 'create a random one sentence story within 10 words'
    while i < 5:
        response = co.generate(model='command',
                                prompt=prompt,
                                max_tokens=150,
                                temperature=0.3,
                                k=0,
                                p=0.95,
                                frequency_penalty=0,
                                presence_penalty=0,
                                stop_sequences=[],
                                return_likelihoods='NONE')
        sentence = response.generations[0].text
        if sentence in sentences:
            prompt = f'create a random one sentence story within 10 words that is different than {sentence}'
        else:
            i += 1
            sentences.append(sentence)
            prompt = 'create a random one sentence story within 10 words'
            image = pipe(sentence).images[0]
            image.save(f'static/generated/generated_image/words_{i}.jpg')

    i=0
    prompt = 'generate one word that is hard to read for dyslexic people'
    while i < 5:
        response = co.generate(model='command',
                                prompt=prompt,
                                max_tokens=150,
                                temperature=0.3,
                                k=0,
                                p=0.95,
                                frequency_penalty=0,
                                presence_penalty=0,
                                stop_sequences=[],
                                return_likelihoods='NONE')
        word = response.generations[0].text
        if word in words:
            prompt = f'generate one word that is hard to read for dyslexic people that is different than {word}'
        else:
            i += 1
            words.append(word)
            prompt = 'generate one word that is hard to read for dyslexic people'
            image = pipe(word).images[0]
            image.save(f'static/generated/generated_image/story_{i}.jpg')

    with open("static/generated/generated_data.py", "a") as f:
        f.write(f"challenging_words={words}" + '\n')
        f.write(f"stories={sentences}")

if __name__ == "__main__":
    generate()
