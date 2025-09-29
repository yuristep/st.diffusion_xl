# Stable Diffusion XL — Генератор изображений

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

CLI-инструмент для генерации изображений через библиотеку `diffusers`. Поддержаны пресеты SDXL и совместимые модели, автоименование файлов и Windows/PowerShell окружение.

## 🚀 Возможности

- 🎨 **Генерация изображений** по текстовым описаниям
- 📁 **Поддержка промптов из файла** с разделением на query и negativePrompt
- ⚡ **Оптимизация для GPU** с автоматическим определением устройства
- 🕒 **Автоматическое именование** файлов по шаблону `{модель}_{YYYYMMDD_HHMMSS}.png`
- ⚙️ **Настраиваемые параметры** генерации
- 🔌 **Поддержка множества моделей** через `main.py` с параметром `--model-key`
- 🌐 **Удаленный инференс** через Hugging Face InferenceClient

## 📋 Поддерживаемые модели

| Ключ модели | Название | Описание | Рекомендуемые шаги |
|-------------|----------|----------|-------------------|
| `sdxl_base` | Stable Diffusion XL Base | Базовая SDXL модель | 25-50 |
| `sdxl_turbo` | SDXL Turbo | Сверхбыстрая генерация | 1-4 |
| `realvisxl` | RealVis XL | Фотореалистичные изображения | 25-50 |
| `juggernautxl` | Juggernaut XL | Универсальная модель | 25-50 |
| `dreamshaperxl` | DreamShaper XL | Художественные изображения | 25-50 |
| `animaginexl` | Animagine XL | Аниме-стиль | 25-50 |
| `ponyxl` | Pony Diffusion XL | Аниме-стиль | 25-50 |
| `zavychromaxl` | ZavyChroma XL | Художественные изображения | 25-50 |
| `sdxl_lightning` | SDXL Lightning | Экстремально быстрая | 1-8 |

## 🛠 Установка

### Требования

- Python 3.8+
- PyTorch с поддержкой CUDA (рекомендуется)
- Hugging Face API токен для доступа к моделям
- Минимум 8GB RAM (16GB рекомендуется)
- Минимум 10GB свободного места на диске

### Пошаговая установка

1. **Клонируйте репозиторий:**
```bash
git clone https://github.com/yuristep/st.diffusion_xl.git
cd st.diffusion_xl
```

2. **Создайте виртуальное окружение (PowerShell):**
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. **Установите зависимости:**
```bash
pip install -r requirements.txt
```

4. **Создайте файл `.env` в корне проекта:**
```env
HF_TOKEN=your_huggingface_token_here
HF_TOKEN=your_huggingface_token_here
```

## 🎯 Использование

### CLI режим (SDXL модели)

#### Базовые команды

```bash
# Базовая SDXL
python main.py --model-key sdxl_base -p "Astronaut in a jungle, realistic photo"

# SDXL Turbo (быстро, 1–4 шага)
python main.py --model-key sdxl_turbo --steps 4 -p "Astronaut in a jungle, realistic photo"

# RealVis XL (фотореализм)
python main.py --model-key realvisxl -p "Astronaut in a jungle, realistic photo"

# SDXL Lightning (по оф. инструкции UNet ckpt)
python main.py --model-key sdxl_lightning --steps 4 -p "Astronaut in a jungle, realistic photo"
```

#### Специализированные модели

```bash
# DreamShaper XL / Animagine XL / Pony / Juggernaut XL / ZavyChroma XL
python main.py --model-key dreamshaperxl -p "Beautiful landscape painting"
python main.py --model-key animaginexl -p "Anime character, detailed"
python main.py --model-key ponyxl -p "Cute pony character"
python main.py --model-key juggernautxl -p "Fantasy warrior"
python main.py --model-key zavychromaxl -p "Artistic portrait"
```

#### Использование файла с промптом

```bash
python main.py --model-key animaginexl --file examples/prompt.txt
```

#### Удаленный инференс (без загрузки модели)

```bash
python main.py --model-key sdxl_base --use-inference --hf-token YOUR_TOKEN -p "Your prompt"
```

### Параметры командной строки

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--model-key` | Ключ модели из списка | `sdxl_base` |
| `--model` | Прямой ID модели Hugging Face | - |
| `-p, --prompt` | Текстовый промпт | - |
| `--file` | Файл с промптом | - |
| `--negative` | Негативный промпт | - |
| `--steps` | Количество шагов | 30 |
| `--guidance` | Guidance scale | 5.5 |
| `--height` | Высота изображения | 768 |
| `--width` | Ширина изображения | 512 |
| `--out` | Путь для сохранения | Автоименование |
| `--use-inference` | Использовать удаленный инференс | False |
| `--hf-token` | Hugging Face токен | Из .env |

## 📁 Формат файла промпта

Создайте файл `examples/prompt.txt` в следующем формате:

```
#query:
Ваш основной промпт для генерации изображения

#negativePrompt:
Что не должно быть на изображении
```

## ⚙️ Параметры конфигурации

### main.py (SDXL модели)

В файле `main.py` значения по умолчанию в словаре `DEFAULTS`:

```python
DEFAULTS = {
    "STEPS": 30,
    "GUIDANCE_SCALE": 5.5,
    "HEIGHT": 768,
    "WIDTH": 512,
}
```

### Настройка через переменные окружения

```env
# В файле .env
HF_TOKEN=your_huggingface_token_here
HF_TOKEN=your_huggingface_token_here
SDXL_MODEL_ID=custom/model/id
```

## 📂 Структура проекта

```
st.diffusion_xl/
├── main.py              # CLI для SDXL/совместимых моделей
├── demo_main.py         # Демонстрационный скрипт
├── requirements.txt     # Зависимости Python
├── README.md            # Документация
├── .env                 # Переменные окружения (создать)
├── .gitignore           # Игнорируемые файлы
└── examples/
    ├── prompt.txt       # Пример файла с промптом
    └── .env             # Альтернативное расположение .env
```

## 💡 Примеры промптов

### Реалистичные изображения
- `"Beautiful sunset over ocean with palm trees, photorealistic"`
- `"Portrait of a woman in Renaissance style, detailed"`
- `"Futuristic city skyline at night, cyberpunk style"`

### Аниме стиль
- `"Anime girl with blue hair, detailed face, studio lighting"`
- `"Cute anime cat character, kawaii style"`
- `"Anime warrior in fantasy armor, dynamic pose"`

### Художественные изображения
- `"Abstract painting with vibrant colors, artistic"`
- `"Digital art of a dragon, fantasy style"`
- `"Watercolor landscape with mountains and lake"`

## 🔧 Устранение неполадок

### Частые проблемы

1. **HF_TOKEN не найден**
   - Убедитесь, что файл `.env` создан и содержит токен Hugging Face
   - Проверьте правильность переменной `HF_TOKEN`

2. **Медленная генерация**
   - Установите PyTorch с поддержкой CUDA
   - Проверьте, что используется GPU: `torch.cuda.is_available()`
   - Используйте модели Turbo или Lightning для быстрой генерации

3. **Нехватка места на диске**
   - Очистите кэш: `~/.cache/huggingface/hub`
   - Перенесите `HF_HOME` на другой диск
   - Используйте удаленный инференс с `--use-inference`

4. **404 при загрузке моделей**
   - Проверьте корректность `repo_id` на Hugging Face
   - Некоторые модели требуют `.bin` вместо `safetensors`
   - Убедитесь в наличии токена для приватных моделей

5. **Ошибки памяти**
   - Уменьшите размер изображения (`--height`, `--width`)
   - Используйте `torch.float16` вместо `torch.float32`
   - Закройте другие приложения

### Логи и отладка

```bash
# Включить подробные логи
python main.py --model-key sdxl_base -p "test" --verbose

# Проверить доступность GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 📊 Производительность

### Рекомендуемые настройки по моделям

| Модель | Шаги | Guidance | Размер | Время (GPU) |
|--------|------|----------|--------|-------------|
| SDXL Base | 30-50 | 5.5-7.5 | 768x512 | ~30-60с |
| SDXL Turbo | 1-4 | 0-1 | 768x512 | ~5-15с |
| SDXL Lightning | 1-8 | 0-1 | 768x512 | ~3-20с |
| RealVis XL | 25-50 | 5.5-7.5 | 768x512 | ~30-60с |

### Оптимизация памяти

- Используйте `torch.float16` для GPU
- Уменьшите размер изображения для экономии памяти
- Используйте модели Turbo/Lightning для быстрой генерации

## 🤝 Вклад в проект

1. Форкните репозиторий
2. Создайте ветку для новой функции (`git checkout -b feature/AmazingFeature`)
3. Зафиксируйте изменения (`git commit -m 'Add some AmazingFeature'`)
4. Отправьте в ветку (`git push origin feature/AmazingFeature`)
5. Откройте Pull Request

## 📄 Лицензия

Этот проект распространяется под лицензией MIT. См. файл `LICENSE` для подробностей.

## 🙏 Благодарности

- [Hugging Face](https://huggingface.co/) за библиотеку `diffusers`
- [Stability AI](https://stability.ai/) за модели Stable Diffusion
- Сообщество разработчиков за вклад в развитие проекта

## 📞 Поддержка

Если у вас возникли вопросы или проблемы:

1. Проверьте раздел [Устранение неполадок](#-устранение-неполадок)
2. Создайте [Issue](https://github.com/yuristep/st.diffusion_xl/issues) на GitHub
3. Обратитесь к [документации Hugging Face](https://huggingface.co/docs/diffusers)

---

**Создано с ❤️ для сообщества разработчиков**