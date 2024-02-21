from io import BytesIO
from pathlib import Path
from modal import Image, Stub, build, enter, gpu, method

##
## Define a container image
##
container = (
    Image.debian_slim(python_version="3.10")
    .apt_install(
        "libglib2.0-0",
        "libsm6",
        "libxrender1",
        "libxext6",
        "ffmpeg",
        "libgl1"
    )
    .pip_install(
        "diffusers~=0.26.2",
        "invisible_watermark~=0.1",
        "transformers~=4.31",
        "accelerate~=0.25.0",
        "safetensors~=0.4.1",
        "compel~=2.0.0"
    )
)

stub = Stub("stable-diffusion-xl", image=container)

with container.imports():
    import torch
    from diffusers import AutoencoderKL, StableDiffusionXLPipeline
    from huggingface_hub import snapshot_download
    from compel import Compel, ReturnedEmbeddingsType


##
## remote process
## Load model and run inference
##
@stub.cls(gpu=gpu.A10G(), container_idle_timeout=240)
class Model:
    ##
    ## download models
    ##
    @build()
    def build(self):
        # Ignore files that we don't need to speed up download time.
        ignore = [
            "*.bin",
            "*.onnx_data",
            "*/diffusion_pytorch_model.safetensors",
        ]
        snapshot_download(
            "cagliostrolab/animagine-xl-3.0", ignore_patterns=ignore
        )
        snapshot_download(
            "madebyollin/sdxl-vae-fp16-fix", ignore_patterns=ignore
        )

    ##
    ## load model
    ##
    @enter()
    def enter(self):
        # options
        load_options = dict(
            torch_dtype=torch.float16,
            use_safetensors=True,
            device_map="auto",
        )
        # load VAE component
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", 
            **load_options
        )
        # load model
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "cagliostrolab/animagine-xl-3.0",
            vae=vae,
            **load_options
        )

        # Compiling the model graph is JIT so this will increase inference time for the first run
        # but speed up subsequent runs. Uncomment to enable.
        # self.base.unet = torch.compile(self.base.unet, mode="reduce-overhead", fullgraph=True)
        # self.refiner.unet = torch.compile(self.refiner.unet, mode="reduce-overhead", fullgraph=True)

    ##
    ## Unlock NSFW
    ##
    @method()
    def null_safety(images, **kwargs):
        return images, False
    
    ##
    ## run inference
    ##
    @method()
    def generate(self, prompt, n_steps=30):
        negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name"
        
        # NSFW setting
        self.pipe.safety_checker = self.null_safety

        # handle token limit
        compel = Compel(tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2] , text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])
        conditioning, pooled = compel(prompt)

        # generate base image
        image = self.pipe(
            # prompt=prompt,
            prompt_embeds=conditioning,
            pooled_prompt_embeds=pooled,
            negative_prompt=negative_prompt,
            width=1024,
            height=1024,
            guidance_scale=7,
            num_inference_steps=n_steps
        ).images[0]

        # write image
        byte_stream = BytesIO()
        image.save(byte_stream, format="PNG")
        image_bytes = byte_stream.getvalue()

        return image_bytes


##
## local process
##
@stub.local_entrypoint()
def main(prompt: str):
    image_bytes = Model().generate.remote(prompt)

    dir = Path("tmp")
    if not dir.exists():
        dir.mkdir(exist_ok=True, parents=True)

    output_path = dir / "output.png"
    print(f"Saving it to {output_path}")
    with open(output_path, "wb") as f:
        f.write(image_bytes)

