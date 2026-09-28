# Экспорт LaMa в ONNX — что нужно сделать вам

Экспорт нельзя выполнить за вас: он требует скачивания весов LaMa и запуска
на вашей машине (в песочнице, где готовился этот код, нет доступа к
huggingface/github и нет вашего процессора для замеров).

Ниже всё, что нужно, — три команды.

## 1. Экспорт и замер

Откройте папку проекта и выполните:

```bat
cd /d C:\путь\к\IOPaint
.venv\Scripts\python -m pip install onnx onnxruntime
.venv\Scripts\python scripts\export_lama_onnx.py --benchmark
```

Скрипт:

1. скачает обычные веса LaMa (те же, что использует программа);
2. сохранит `lama.onnx` в папку моделей (`<IOPAINT_MODEL_DIR>\erase_models`);
3. замерит скорость PyTorch vs ONNX Runtime на вашем Ryzen 5700G.

Первый запуск занимает 5–15 минут (в основном скачивание и трассировка).
Нужно ~2 ГБ свободного места и лучше закрыть тяжёлые программы: экспорт
временно ест 4–6 ГБ RAM.

## 2. Если хотите проверить ещё и встроенную видеокарту

```bat
.venv\Scripts\python -m pip install onnxruntime-directml
.venv\Scripts\python scripts\export_lama_onnx.py --benchmark
```

Тогда в отчёте появится строка `DmlExecutionProvider` — это ваша Radeon Vega.

## 3. Что прислать

Скопируйте блок между строками `----- copy everything below/above this line -----`.
Он выглядит примерно так:

```
machine   : AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD
python    : 3.12.6
torch     : 2.5.1
onnxruntime: 1.20.1 ['DmlExecutionProvider', 'CPUExecutionProvider']
size      : 1024x1024, runs: 5
pytorch-cpu: 4200.0 ms
CPUExecutionProvider: 2600.0 ms
DmlExecutionProvider: 1500.0 ms
```

Также пригодятся:

* полный лог, если скрипт упал (ошибка при трассировке — обычное дело для
  конкретной версии torch, по тексту видно, что чинить);
* размер полученного `lama.onnx` (обычно ~200 МБ).

По этим цифрам подключается ONNX-бэкенд в качестве варианта по умолчанию:
если DirectML быстрее — стирание пойдёт через iGPU, если нет — останется CPU,
но с ONNX Runtime, который на AMD обычно на 30–80% быстрее PyTorch.

## Частые вопросы

**«Это обязательно?»** Нет. Программа полностью работает и без ONNX,
экспорт — про скорость.

**«Можно ли сломать установку?»** Нет: скрипт только читает веса и пишет
один новый файл. Удалить его можно `windows\clean-cache.bat` или вручную.

**«Ошибка `torch.onnx.export ... unsupported op`»** — пришлите лог,
понадобится либо другая опция `--opset` (пробуйте `--opset 16`), либо
патч одного слоя LaMa. Это ожидаемый сценарий, не ваша вина.
