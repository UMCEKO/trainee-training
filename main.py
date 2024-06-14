import random

import PIL.Image
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import torch
from functools import cache

@cache
def load_model():
    pipe = StableDiffusionXLPipeline.from_pretrained(
        'SG161222/RealVisXL_V4.0',
        torch_dtype=torch.float16,
    )
    pipe.to('cuda')
    pipe.enable_xformers_memory_efficient_attention()
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True,
                                                             algorithm_type="sde-dpmsolver++")
    return pipe


while True:
    seed = -1

    if seed == -1:
        seed = random.randint(1, 10 ** 10)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    pipe = load_model()
    output = pipe(
        prompt=input('Enter a prompt: '),
        negative_prompt='stars',
        num_inference_steps=30,
        generator=generator,
        width=768,
        height=1280,
    )
    image: PIL.Image.Image = output.images[0]
    image.show()