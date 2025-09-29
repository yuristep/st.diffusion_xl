import os
import re
from datetime import datetime
from pathlib import Path
import argparse

from dotenv import load_dotenv
from typing import Optional
import torch


# Загрузка переменных окружения из .env в корне либо из examples/.env
if Path(".env").exists():
    load_dotenv(".env")
elif Path("examples/.env").exists():
    load_dotenv("examples/.env")


# Единые значения по умолчанию 
DEFAULTS = {
    "STEPS": 30,
    "GUIDANCE_SCALE": 5.5,
    "HEIGHT": 768,
    "WIDTH": 512,
}


class Config:
    # Модель SDXL (RealVisXL). Можно заменить на другую совместимую SDXL.
    MODEL_ID = os.getenv("SDXL_MODEL_ID", "SG161222/RealVisXL_V4.0")
    DEFAULT_STEPS = DEFAULTS["STEPS"]
    DEFAULT_GUIDANCE_SCALE = DEFAULTS["GUIDANCE_SCALE"]
    DEFAULT_HEIGHT = DEFAULTS["HEIGHT"]
    DEFAULT_WIDTH = DEFAULTS["WIDTH"]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


MODEL_PRESETS = {
    # Базовая SDXL от Stability AI
    "sdxl_base": {
        "id": "stabilityai/stable-diffusion-xl-base-1.0",
        "pipeline": "StableDiffusionXLPipeline",
        "variant": "fp16",
        "use_safetensors": True,
        "steps_hint": (25, 50),
    },
    # SDXL Turbo — сверхбыстрая (1-4 шага)
    "sdxl_turbo": {
        "id": "stabilityai/sdxl-turbo",
        "pipeline": "DiffusionPipeline",
        "variant": "fp16",
        "use_safetensors": True,
        "steps_hint": (1, 4),
    },
    # RealVis XL — фотореализм
    "realvisxl": {
        "id": "SG161222/RealVisXL_V4.0",
        "pipeline": "StableDiffusionXLPipeline",
        "variant": "fp16",
        "use_safetensors": True,
        "steps_hint": (20, 40),
    },
    # Juggernaut XL — универсальная
    "juggernautxl": {
        "id": "RunDiffusion/Juggernaut-XL",
        "pipeline": "DiffusionPipeline",
        "variant": "fp16",
        "use_safetensors": False,
        "steps_hint": (20, 40),
    },
    # DreamShaper XL — художественная
    "dreamshaperxl": {
        "id": "lykon/dreamshaper-xl-1-0",
        "pipeline": "DiffusionPipeline",
        "variant": "fp16",
        "use_safetensors": True,
        "steps_hint": (20, 40),
    },
    # Animagine XL — аниме/стилизованная
    "animaginexl": {
        "id": "cagliostrolab/animagine-xl-3.0",
        "pipeline": "DiffusionPipeline",
        "variant": "fp16",
        "use_safetensors": True,
        "steps_hint": (20, 40),
    },
    # Pony Diffusion XL — аниме
    "ponyxl": {
        "id": "stabilityai/stable-diffusion-xl-base-1.0",
        "pipeline": "DiffusionPipeline",
        "variant": "fp16",
        "use_safetensors": True,
        "steps_hint": (20, 40),
    },
    # ZavyChroma XL — художественная
    "zavychromaxl": {
        "id": "stablediffusionapi/zavychromaxl",
        "pipeline": "DiffusionPipeline",
        "variant": None,
        "use_safetensors": False,
        "steps_hint": (20, 40),
    },
    # SDXL Lightning — сверхбыстрая
    "sdxl_lightning": {
        "id": "ByteDance/SDXL-Lightning",
        "pipeline": "DiffusionPipeline",
        "variant": "fp16",
        "use_safetensors": True,
        "steps_hint": (1, 8),
    },
    # (Flux Schnell удалён из пресетов как нестабильный в этом окружении)
}


def _resolve_pipeline(pipeline_name: str):
    if pipeline_name == "StableDiffusionXLPipeline":
        from diffusers import StableDiffusionXLPipeline as P
        return P
    if pipeline_name == "DiffusionPipeline":
        from diffusers import DiffusionPipeline as P
        return P
    if pipeline_name == "FluxPipeline":
        from diffusers import FluxPipeline as P
        return P
    # По умолчанию пробуем универсальный
    from diffusers import DiffusionPipeline as P
    return P


def initialize_model(model_id_or_key: str, num_steps: Optional[int] = None):
    # Разрешаем как явный ID, так и ключ пресета
    preset = MODEL_PRESETS.get(model_id_or_key)
    if preset:
        model_id = preset["id"]
        pipeline_name = preset["pipeline"]
        variant = preset.get("variant")
        use_safetensors = preset.get("use_safetensors", True)
    else:
        model_id = model_id_or_key
        # Эвристика: SDXL → StableDiffusionXLPipeline, иначе универсальный
        pipeline_name = "StableDiffusionXLPipeline" if "xl" in model_id.lower() else "DiffusionPipeline"
        variant = "fp16"
        use_safetensors = True

    # Специальная сборка для SDXL Lightning: базовая SDXL + UNet из ByteDance/SDXL-Lightning
    if model_id_or_key == "sdxl_lightning":
        # Реализация по официальной документации ByteDance/SDXL-Lightning
        # https://huggingface.co/ByteDance/SDXL-Lightning
        from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        print("\n🔄 Загружаем SDXL Lightning по инструкции (base + UNet ckpt)...")
        steps = num_steps or 4
        if steps not in {1, 2, 4, 8}:
            print(f"⚠️ SDXL Lightning поддерживает шаги 1/2/4/8. Использую 4.")
            steps = 4

        base_id = "stabilityai/stable-diffusion-xl-base-1.0"
        device = Config.DEVICE
        dtype = Config.TORCH_DTYPE

        # 1) Создаём UNet по конфигу базовой SDXL и загружаем веса lightning ckpt
        unet = UNet2DConditionModel.from_config(base_id, subfolder="unet").to(device, dtype)

        if steps == 1:
            ckpt = "sdxl_lightning_1step_unet_x0.safetensors"
        else:
            ckpt = f"sdxl_lightning_{steps}step_unet.safetensors"

        print(f"🔎 Загружаю веса UNet: ByteDance/SDXL-Lightning/{ckpt}")
        state = load_file(hf_hub_download("ByteDance/SDXL-Lightning", ckpt), device=device)
        unet.load_state_dict(state)

        # 2) Собираем пайплайн с заменённым UNet
        pipe = StableDiffusionXLPipeline.from_pretrained(
            base_id,
            unet=unet,
            torch_dtype=dtype,
            variant=(variant if device == "cuda" else None),
        ).to(device)

        # 3) Настраиваем планировщик по рекомендации (trailing). Для 1 шага — prediction_type="sample"
        if steps == 1:
            pipe.scheduler = EulerDiscreteScheduler.from_config(
                pipe.scheduler.config, timestep_spacing="trailing", prediction_type="sample"
            )
        else:
            pipe.scheduler = EulerDiscreteScheduler.from_config(
                pipe.scheduler.config, timestep_spacing="trailing"
            )
    else:
        P = _resolve_pipeline(pipeline_name)
        print("\n🔄 Загружаем модель SDXL/Flux...")
        pipe = P.from_pretrained(
            model_id,
            torch_dtype=Config.TORCH_DTYPE,
            variant=(variant if Config.DEVICE == "cuda" else None),
            use_safetensors=use_safetensors,
        )
    pipe = pipe.to(Config.DEVICE)

    # Лёгкие оптимизации для GPU
    if Config.DEVICE == "cuda":
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pass

    print("✅ SDXL успешно загружена!")
    return pipe


def truncate_prompt_to_77_tokens(pipe, text: str) -> str:
    """Больше не обрезает промпт. Если длина > 77 токенов, выводит информационное сообщение.
    Реальная максимальная длина для CLIP (текстового энкодера) — ~77 токенов.
    """
    try:
        if not text:
            return text
        tokenizer = getattr(pipe, "tokenizer", None) or getattr(pipe, "tokenizer_2", None)
        if tokenizer is not None:
            encoded = tokenizer(
                text,
                truncation=False,
                return_overflowing_tokens=False,
                return_attention_mask=False,
                add_special_tokens=True,
            )
            input_ids = encoded.get("input_ids", [])
            if input_ids and len(input_ids[0]) > 77:
                print("ℹ️ Если требуется максимально точное соответствие промпта изображению, смысл лучше формулировать коротко и ёмко, не превышая 77 токенов")
        else:
            # Грубая эвристика по количеству слов
            words = re.split(r"\s+", text.strip())
            if len(words) > 77:
                print("ℹ️ Если требуется максимально точное соответствие промпта изображению, смысл лучше формулировать коротко и ёмко, не превышая 77 токенов")
        return text
    except Exception:
        return text


def generate_image_sdxl(
    pipe,
    prompt: str,
    negative_prompt: str = None,
    num_inference_steps: int = None,
    guidance_scale: float = None,
    height: int = None,
    width: int = None,
    output_path: str = None,
):
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"sdxl_{timestamp}.png"

    if num_inference_steps is None:
        num_inference_steps = Config.DEFAULT_STEPS
    if guidance_scale is None:
        guidance_scale = Config.DEFAULT_GUIDANCE_SCALE
    if height is None:
        height = Config.DEFAULT_HEIGHT
    if width is None:
        width = Config.DEFAULT_WIDTH

    # Безопасное усечение промпта до 77 токенов
    safe_prompt = truncate_prompt_to_77_tokens(pipe, prompt)
    safe_negative = truncate_prompt_to_77_tokens(pipe, negative_prompt) if negative_prompt else None

    print("\n🎨 SDXL генерация...")
    print(f"📝 Промпт: {safe_prompt[:100]}{'...' if len(safe_prompt) > 100 else ''}")
    if safe_negative:
        print(f"🚫 Негативный: {safe_negative[:100]}{'...' if len(safe_negative) > 100 else ''}")
    print(f"⚙️ Параметры: {num_inference_steps} шагов, guidance={guidance_scale}, размер={width}x{height}")

    result = pipe(
        prompt=safe_prompt,
        negative_prompt=safe_negative,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
    )
    image = result.images[0]
    image.save(output_path)
    print(f"✅ Сохранено: {output_path}")
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Генерация изображений с SDXL/Flux через diffusers",
        add_help=True,
    )
    parser.add_argument(
        "--model",
        default=Config.MODEL_ID,
        help="ID модели на Hugging Face (если указан --model-key, то он имеет приоритет)",
    )
    parser.add_argument(
        "--model-key",
        choices=sorted(MODEL_PRESETS.keys()),
        help="Ключ пресета модели (напр. sdxl_base, sdxl_turbo, realvisxl, juggernautxl, ...)",
    )
    parser.add_argument("--prompt", "-p", default=None, help="Текстовый промпт")
    parser.add_argument("--negative", "-n", default=None, help="Негативный промпт")
    parser.add_argument("--steps", type=int, default=Config.DEFAULT_STEPS, help="Число шагов диффузии")
    parser.add_argument("--guidance", type=float, default=Config.DEFAULT_GUIDANCE_SCALE, help="Guidance scale")
    parser.add_argument("--height", type=int, default=Config.DEFAULT_HEIGHT, help="Высота изображения")
    parser.add_argument("--width", type=int, default=Config.DEFAULT_WIDTH, help="Ширина изображения")
    parser.add_argument("--out", "-o", default=None, help="Путь для сохранения PNG")
    parser.add_argument("--file", "-f", default=None, help="Путь к текстовому файлу с промптом")
    # Удалённый инференс через Hugging Face Inference API
    parser.add_argument("--use-inference", action="store_true", help="Использовать Hugging Face InferenceClient вместо локальной модели")
    parser.add_argument("--inference-provider", default="fal-ai", help="Провайдер для InferenceClient (по умолчанию fal-ai)")
    parser.add_argument("--hf-token", default=None, help="HF_TOKEN, иначе возьмётся из переменных окружения")
    # LoRA для SDXL Lightning
    # LoRA функционал временно убран
    return parser


def parse_prompt_file(file_path: Path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        query_match = re.search(r"#query:\s*\n(.*?)(?=\n#|$)", content, re.DOTALL | re.IGNORECASE)
        negative_match = re.search(r"#negativePrompt:\s*\n(.*?)(?=\n#|$)", content, re.DOTALL | re.IGNORECASE)
        prompt = query_match.group(1).strip() if query_match else content.strip()
        negative_prompt = negative_match.group(1).strip() if negative_match else None
        return prompt, negative_prompt
    except Exception as e:
        print(f"❌ Ошибка при чтении файла {file_path}: {e}")
        return None, None


def _model_slug_for_filename(model_id_or_key: str) -> str:
    preset = MODEL_PRESETS.get(model_id_or_key)
    if preset and preset.get("id"):
        source = preset["id"]
    else:
        source = model_id_or_key
    # Берём хвост после '/', приводим к нижнему регистру и заменяем не-алфанум на '_'
    name = source.split("/")[-1]
    name = name.lower()
    import re as _re
    name = _re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    return name or "model"


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # Если просто спрашивают справку, сюда не дойдём — argparse сам её покажет и выйдет

    # Получение промпта: из аргумента или из файла
    prompt = args.prompt
    negative = args.negative
    if not prompt and args.file:
        p, n = parse_prompt_file(Path(args.file))
        prompt = prompt or p
        negative = negative or n

    if not prompt:
        print("⚠️ Не задан промпт. Укажите --prompt или --file.")
        return

    # Удалённый инференс через Hugging Face InferenceClient
    if args.use_inference:
        try:
            from huggingface_hub import InferenceClient
        except Exception:
            print("❌ huggingface_hub не установлен. Установите пакет и повторите.")
            return

        token = args.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not token:
            print("❌ Требуется HF_TOKEN (аргумент --hf-token или переменная окружения HF_TOKEN).")
            return

        provider = args.inference_provider
        model_for_inference = args.model if args.model else MODEL_PRESETS.get(args.model_key or "", {}).get("id", "ByteDance/SDXL-Lightning")

        print(f"\n☁️ Запуск удалённого инференса: provider={provider}, model={model_for_inference}")
        client = InferenceClient(provider=provider, api_key=token)
        image = client.text_to_image(
            args.prompt,
            model=model_for_inference,
        )
        if args.out:
            out_path = args.out
        else:
            model_slug = _model_slug_for_filename(args.model_key or args.model or model_for_inference)
            out_path = f"{model_slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        image.save(out_path)
        print(f"✅ Сохранено: {out_path}")
        return

    # Локальный режим
    model_key = args.model_key
    model_id_or_key = model_key if model_key else args.model

    preset = MODEL_PRESETS.get(model_id_or_key)
    if preset and preset.get("steps_hint"):
        min_s, max_s = preset["steps_hint"]
        if args.steps < min_s or args.steps > max_s:
            print(f"ℹ️ Рекомендуемое число шагов для {model_id_or_key}: {min_s}-{max_s}. Продолжаем с {args.steps}.")

    pipe = initialize_model(model_id_or_key, num_steps=args.steps)

    # LoRA отключена

    # Информация об устройстве
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Устройство: {device}")
    if device == "cuda":
        try:
            print(f"🎮 GPU: {torch.cuda.get_device_name()}")
        except Exception:
            pass

    output_path = args.out
    if output_path is None:
        model_slug = _model_slug_for_filename(model_id_or_key)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{model_slug}_{timestamp}.png"
    generate_image_sdxl(
        pipe,
        prompt=prompt,
        negative_prompt=negative,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        height=args.height,
        width=args.width,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()


