# SR-Inpainting — лабораторная работа №3

Восстановление изображений ×4 с использованием `stabilityai/stable-diffusion-x4-upscaler` и известных фрагментов оригинального HR. Веса модели не изменяются. LR остаётся условием U-Net на каждом шаге.

## 1. Теория и алгоритм

- `LR`: RGB-изображение H×W, `HR`: RGB-изображение 4H×4W.
- Маска M: бинарный массив H×W. **1 — генерировать, 0 — сохранить HR**.
- VAE модели сжимает HR в латенты 4×H×W (коэффициент масштабирования из конфигурации VAE: 0.08333).
- U-Net принимает 4 канала латентов и 3 канала зашумлённого LR, а также время, текстовое условие и noise level. Сохраняется штатное `v_prediction` модели.

Перед кодированием неизвестные пиксели HR заменяются нулями в RGB, чтобы они не попадали в латенты известных областей через рецептивное поле VAE. Используется детерминированная мода распределения энкодера. Далее известные латенты `z_hr` зашумляются до текущего временного шага основным DDIM Scheduler:

```text
z_hr(t) = sqrt(alpha_bar(t)) * z_hr + sqrt(1 - alpha_bar(t)) * epsilon_hr
z(t)    = where(M, z_diff(t), z_hr(t))
```

Проекция применяется до первого вызова U-Net и после каждого `scheduler.step`, с зашумлением HR до **следующего** шага из расписания DDIM. На последнем шаге подставляются чистые известные латенты. После декодирования выполняется точное RGB-смешивание:

```text
SR = where(M_HR, generated, HR)
```

M_HR получается повторением каждого элемента маски 4×4 (nearest). LR не смешивается с результатом диффузии. Известные пиксели HR копируются без потерь; VAE не обязан воспроизводить их идеально.

Вместо копирования всего `StableDiffusionUpscalePipeline` используется компактная обёртка и штатный callback Diffusers 0.35.2. Callback меняет живой тензор латентов **in-place** после каждого шага DDIM (`copy_`); возвращаемое значение callback не используется. DDIM имеет `order=1`, `init_noise_sigma=1`, `eta=0`; алгоритм не предназначен для произвольных scheduler без адаптации. Для каждой генерации используются отдельные воспроизводимые CPU RNG: seed для начального шума, seed+1 для шума LR, seed+2 для HR. VAE работает в fp32 для предотвращения переполнений.

Граничные случаи: M=1 использует обычный SR без кодирования HR; M=0 возвращает исходный HR без диффузии (в CLI даже без загрузки модели). Это намеренная оптимизация.

## 2. Установка и архитектура

Python 3.10–3.12; рекомендуется 3.11. macOS Apple Silicon / MPS, CUDA или CPU. Для практической работы желательно GPU и достаточный объём памяти (ориентир — 16 ГБ VRAM/объединённой памяти или больше; это не измеренное минимальное требование). Начать можно с LR 64×64 или 128×128; при нехватке памяти уменьшить размер. Веса модели и метрик скачиваются при первом запуске, требуется интернет и несколько ГБ свободного места в кэше Hugging Face/PyTorch. CPU существенно медленнее.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

На Windows: `.venv\Scripts\activate`. Для CUDA при необходимости установите совместимую сборку torch/torchvision по [инструкции PyTorch](https://pytorch.org/get-started/locally/) перед установкой остальных зависимостей. На macOS можно включить fallback неподдерживаемых MPS-операций до запуска Python:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

```text
SR-Inpainting/
├── main.py                         # одиночный запуск
├── requirements.txt
├── src/
│   ├── pipeline.py                 # модификация сэмплинга
│   ├── degradation.py               # синтетические деградации
│   ├── masks.py                    # маски LR и HR
│   ├── metrics.py                  # PSNR, LPIPS, CLIP-IQA
│   └── utils.py                    # загрузка, валидация, смешивание
├── experiments/
│   ├── run_baseline.py             # обычный SR
│   └── run_experiments.py          # сравнение методов, CSV, графики
├── tests/test_project.py           # тесты без загрузки весов
├── data/{HR,LR,masks}/             # входные данные
└── results/{images,plots}/         # результаты
```

### Одиночный запуск

Положите HR-изображение 512×512 в `data/HR/0801.png` (либо укажите собственный путь). Если `--lr` не указан, LR генерируется из HR. LR должен иметь размеры не менее 32 и кратные 8 по каждой оси; HR — строго в 4 раза больше. Одиночный CLI не кадрирует и не растягивает входы.

```bash
# Обычный SR, только LR (файл должен существовать)
python -m experiments.run_baseline --lr data/LR/0801.png --device mps \
  --output results/images/baseline.png

# SR-inpainting: сильная деградация, половина HR известна
python main.py --hr data/HR/0801.png --degradation realesrgan_like \
  --mask known50 --device mps --output results/images/inpaint.png

# SR + финальное смешивание: контрольный метод
python main.py --hr data/HR/0801.png --degradation realesrgan_like \
  --mask known50 --device mps --method composite \
  --output results/images/composite.png

# Внешняя пара HR/LR и собственная бинарная маска PNG или NPY
python main.py --hr data/HR/0801.png --lr data/LR/0801.png \
  --mask data/masks/mask.png --device mps
```

`--device auto` по умолчанию выбирает CUDA → MPS → CPU. `--precision auto` использует fp16 для U-Net/текстового энкодера на GPU, fp32 на CPU; `--precision fp32` — вариант при нестабильности fp16. Общие параметры: `--steps 50 --noise-level 20 --guidance-scale 1 --prompt "" --seed 42`. Для быстрого smoke test можно уменьшить число шагов, но не сравнивать его с полноценным экспериментом. При `guidance-scale > 1` работает стандартный CFG.

Маски: `all_one` (0% известного HR), `all_zero` (100%), `known25`, `known50`, `known75` (известная левая полоса), `center` (генерируется центральный прямоугольник). Внешняя маска строго LR-размера, белый/1 — генерация, чёрный/0 — сохранение. PNG допускает 0/255, NPY — 0/1 или 0/255; промежуточные значения запрещены.

### Пакетный эксперимент

Скачайте HR из [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) и поместите минимум несколько изображений в `data/HR/`. Можно использовать и свои изображения. Скрипт по умолчанию выбирает до 5 файлов в алфавитном порядке и вырезает центральный HR-фрагмент 512×512 (LR 128×128), не растягивая изображения. Исходники должны быть не меньше указанного размера.

```bash
# Быстрая проверка: 1 картинка, 1 деградация, 3 маски, только PSNR
python -m experiments.run_experiments --device mps --size 64 --limit 1 \
  --steps 5 --degradations realesrgan_like \
  --masks all_one all_zero known50 --psnr-only --output results/smoke

# Основной эксперимент (по умолчанию 5 изображений, 3 деградации, 5 масок)
python -m experiments.run_experiments --device mps

# Повтор с несколькими seed
python -m experiments.run_experiments --device mps --seeds 42 43 44 \
  --output results/multiseed
```

Деградации: `bicubic` (×4), `unknown` (синтетическое размытие и случайный способ downsampling), `realesrgan_like` (двухстадийная комбинация blur/resize/noise/JPEG, вдохновлённая Real-ESRGAN). Последние две **не являются** точными официальными реализациями DIV2K unknown и Real-ESRGAN. Seed деградации (`--degradation-seed 42`) отделён от seed генерации, поэтому LR не меняется между повторами.

Для официальных LR DIV2K или собственных пар предусмотрен режим `--paired NAME=DIR`. Имена файлов LR: `<stem>x4.png`, `<stem>.png` или имя исходного HR. Предполагается геометрическое соответствие ×4 и возможный modcrop HR справа/снизу до 3 пикселей. Центральный LR-crop и соответствующий HR-crop вырезаются совместно. Директории ниже приведены как пример, их нужно заполнить скачанными данными:

```bash
python -m experiments.run_experiments --device mps \
  --degradations div2k_bicubic div2k_unknown realesrgan_like \
  --paired div2k_bicubic=data/LR/bicubic \
  --paired div2k_unknown=data/LR/unknown
```

Аналогично можно подключить заранее полученные LR от официального degradation pipeline Real-ESRGAN. Скрипт не запускает сам Real-ESRGAN.

Для каждого изображения, деградации, seed и маски сравниваются:

| Метод | Известный HR при сэмплинге | Финальное RGB-смешивание |
|---|---|---|
| baseline | нет | нет |
| composite | нет | да |
| inpaint | да | да |

Baseline вычисляется только один раз для данной пары и seed, затем переиспользуется во всех масках. При полностью единичной маске используется тот же baseline, при полностью нулевой возвращается HR. Для каждого смешанного варианта используется тот же seed, начальный шум в неизвестных областях, LR, шум LR, prompt и настройки модели.

## 3. Метрики и результаты

- PSNR: RGB, диапазон 0–255, больше — лучше. PSNR_mask вычисляется только по пикселям M=1. Для идеально совпадающих изображений — `inf`, пустая область — `nan`.
- LPIPS: AlexNet, вход [0,1] с `normalize=True`, меньше — лучше.
- CLIP-IQA: оригинальная модель `clip_iqa` через TorchMetrics/PIQ, prompt quality, диапазон [0,1], больше — лучше. Не требует эталона.
- LPIPS_mask и CLIP-IQA_mask: вычисляются на прямоугольном crop восстанавливаемой области без закрашивания остальных пикселей. Для непрямоугольных, пустых или слишком маленьких областей (сторона <64 пикселей) возвращается `nan`. PSNR_mask поддерживает произвольные маски. По умолчанию нейросетевые метрики выполняются на CPU, чтобы не конкурировать с генерацией за память GPU; `--metric-device cuda` или `mps` опциональны.

Файлы после запуска:

```text
results/
├── config.json
├── metrics.csv                   # все методы, изображения, seed, маски
├── summary.csv                   # средние по методу, маске и деградации
├── plots/                        # графики метрик от % известного HR
└── images/<image>/<degradation>/<seed>/
    ├── hr.png
    ├── lr.png
    ├── baseline.png
    └── <mask>/
        ├── mask.png
        ├── composite.png
        ├── inpaint.png
        └── comparison.png        # HR, LR, маска и три метода
```

`nan` исключается из средних; бесконечный PSNR сохраняется в CSV, но не отображается на графике. Графики по доле известного HR строятся для полосовых масок и крайних случаев; `center` не смешивается с ними из-за другой геометрии. `--psnr-only` отключает загрузку моделей LPIPS и CLIP-IQA и оставляет их значения `nan`.

Сравнивать `inpaint` и `composite` нужно прежде всего по **одной и той же неизвестной области**: повышение PSNR всего изображения при увеличении известной доли может быть простым следствием копирования HR. PSNR_mask у baseline и composite обязан совпадать. Более высокое качество inpaint относительно composite покажет эффект HR именно в процессе диффузии. CLIP-IQA может оценить красивый, но не соответствующий эталону результат, поэтому анализируются все метрики и изображения.

## 4. Проверки и выводы

```bash
python -m pytest -q
```

Тесты без скачивания весов: воспроизводимость деградаций, маски и точное RGB-смешивание, метрики и граничные случаи, геометрия пар, подстановка HR на каждом шаге и отсутствие утечки неизвестного HR при фиксированном LR (с тестовым pipeline), пакетный экспорт CSV/графиков. В среде сборки прошли 5 тестов на CPU. Реальный inference Stable Diffusion и LPIPS/CLIP-IQA с весами в этой среде не выполнялись (нет Diffusers и CUDA/MPS). Поэтому численные результаты, производительность и совместимость с конкретным устройством ещё необходимо подтвердить запуском. Отчёт намеренно не содержит вымышленных экспериментальных метрик.

Гипотеза: для сильной деградации LR известные фрагменты HR могут улучшать структурное и визуальное качество неизвестных областей. Улучшение не гарантируется: модель обучалась на SR, а не на таком способе инпэинтинга; маскирование до VAE и его рецептивное поле могут создавать пограничные артефакты. Финальное RGB-смешивание обеспечивает точность известных пикселей, но не гарантирует отсутствие видимых швов. После реального эксперимента добавьте в README таблицы `summary.csv`, сравнения, графики и вывод о гипотезе.

## 5. Источники

- [Stable Diffusion x4 Upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler)
- [Исходный код StableDiffusionUpscalePipeline v0.35.2](https://github.com/huggingface/diffusers/blob/v0.35.2/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_upscale.py)
- [Конфигурация DDIM модели](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler/blob/main/scheduler/scheduler_config.json)
- [Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- [RePaint](https://arxiv.org/abs/2201.09865) — идея согласования известных областей на шагах диффузии; здесь не реализуется полный resampling RePaint.
- [Real-ESRGAN](https://arxiv.org/abs/2107.10833)
- [LPIPS](https://arxiv.org/abs/1801.03924), [CLIP-IQA](https://arxiv.org/abs/2207.12396)
- [TorchMetrics CLIP-IQA](https://lightning.ai/docs/torchmetrics/stable/multimodal/clip_iqa.html)
