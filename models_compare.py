# дообучение нескольких моделей на CIFAR10
# сравнение точности и скорости

import os
from datetime import datetime
import time
import csv
import torch
from torch.utils.data import DataLoader, Subset
from torch import nn
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torchvision import datasets
from sklearn.metrics import classification_report, accuracy_score
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}")

# ----- общие параметры -----
data_dir = 'cifar-10-python'
weights_dir = 'weights'
results_csv = 'results.csv'
val_size = 5000
batch_size = 32
EPOCHS = 3
SEED = 42
NUM_WORKERS = 16
os.makedirs(weights_dir, exist_ok=True)

# ----- ТЕСТОВЫЙ РЕЖИМ -----
# True - быстрая проверка работоспособности (1 эпоха, 5 батчей, ограниченный тест)
# False - полноценный прогон
TEST_MODE = False

if TEST_MODE:
    print("!!! TEST_MODE включён: 1 эпоха, по 5 батчам train/val, инференс на 5 батчах")
    EPOCHS = 1


# ----- timm-модели -----
TIMM_MODELS = {
    'maxvit_base_tf_224.in1k',
    'maxvit_large_tf_224.in1k',
}

def is_timm_model(name):
    return name in TIMM_MODELS


# ----- фиксированный split -----
_raw_train = datasets.CIFAR10(root=data_dir, train=True, download=True)
_raw_test = datasets.CIFAR10(root=data_dir, train=False, download=True)
CLASS_NAMES = _raw_train.classes

g = torch.Generator().manual_seed(SEED)
all_indices = torch.randperm(len(_raw_train), generator=g).tolist()
val_indices = all_indices[:val_size]
train_indices = all_indices[val_size:]
test_indices = list(range(len(_raw_test)))

print(f"Split зафиксирован: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")


def _make_loaders_with_transform(transform):
    """Общая функция: создаёт загрузчики с заданным transform."""
    full_train = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transform)
    full_test = datasets.CIFAR10(root=data_dir, train=False, download=False, transform=transform)

    train_ds = Subset(full_train, train_indices)
    val_ds = Subset(full_train, val_indices)
    test_ds = Subset(full_test, test_indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader, test_loader, len(train_ds), len(val_ds), len(test_ds)


def make_loaders_hf(processor):
    """Загрузчики для HuggingFace-моделей."""
    def img_transform(image):
        inputs = processor(images=image, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0)
    return _make_loaders_with_transform(img_transform)


def make_loaders_timm(model):
    """Загрузчики для timm-моделей."""
    data_cfg = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_cfg, is_training=False)
    return _make_loaders_with_transform(transform)


def safe_name(model_name):
    return model_name.replace('/', '_')


def _setup_model(model_name):
    """
    Возвращает кортеж: (model, train_loader, val_loader, test_loader,
                       train_size, test_size, forward_logits, trainable_params)
    Учитывает разницу между HuggingFace и timm моделями.
    """
    if is_timm_model(model_name):
        model = timm.create_model(model_name, pretrained=True, num_classes=len(CLASS_NAMES))
        model.to(device)
        train_loader, val_loader, test_loader, train_size, _, test_size = make_loaders_timm(model)

        # Заморозка всего, кроме head.*
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith('head.')
        trainable_params = [p for n, p in model.named_parameters() if n.startswith('head.')]

        def forward_logits(inputs):
            return model(inputs)
    else:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(model_name)
        train_loader, val_loader, test_loader, train_size, _, test_size = make_loaders_hf(processor)

        # Подмена головы под 10 классов
        model.classifier = nn.Linear(model.classifier.in_features, len(CLASS_NAMES))
        model.to(device)

        # Заморозка всего, кроме classifier
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
        trainable_params = list(model.classifier.parameters())

        def forward_logits(inputs):
            return model(inputs).logits

    return (model, train_loader, val_loader, test_loader,
            train_size, test_size, forward_logits, trainable_params)


def _setup_model_for_eval(model_name):
    """То же, но без претрена и с загрузкой сохранённых весов."""
    weights_path = f'{weights_dir}/{safe_name(model_name)}_best.pt'

    if is_timm_model(model_name):
        model = timm.create_model(model_name, pretrained=False, num_classes=len(CLASS_NAMES))
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.to(device)
        model.eval()
        _, _, test_loader, _, _, test_size = make_loaders_timm(model)

        def forward_logits(inputs):
            return model(inputs)
    else:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(model_name)
        model.classifier = nn.Linear(model.classifier.in_features, len(CLASS_NAMES))
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.to(device)
        model.eval()
        _, _, test_loader, _, _, test_size = make_loaders_hf(processor)

        def forward_logits(inputs):
            return model(inputs).logits

    return model, test_loader, test_size, forward_logits


def train_pipeline(model_name):
    """Обучает модель, сохраняет лучшие веса. Возвращает время обучения в секундах."""
    print(f'\n#########\n Обучение: {model_name} \n#########\n')

    (model, train_loader, val_loader, _,
     train_size, _, forward_logits, trainable_params) = _setup_model(model_name)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(trainable_params, lr=0.001)
    best_vloss = float('inf')
    best_path = f'{weights_dir}/{safe_name(model_name)}_best.pt'

    if os.path.isfile(best_path):
        currenttime = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = f"{best_path}.{currenttime}.bak"
        os.rename(best_path, backup_path)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    train_start = time.perf_counter()

    for epoch in range(EPOCHS):
        print(f'Эпоха {epoch+1}/{EPOCHS}')
        model.train(True)

        running_loss = 0.0
        for batch_index, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = forward_logits(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_index % 400 == 399:
                avg = running_loss / 400.
                print(f'  батч: {batch_index+1}/{int(train_size/batch_size)+1}, train_loss {avg:.4f}')
                running_loss = 0.

            if TEST_MODE and batch_index >= 4:
                print(f'  [TEST_MODE] обработано 5 батчей train, выходим')
                break

        # валидация
        model.eval()
        running_vloss = 0.0
        n_val_seen = 0
        with torch.no_grad():
            for vbatch_idx, (vinputs, vlabels) in enumerate(val_loader):
                vinputs = vinputs.to(device, non_blocking=True)
                vlabels = vlabels.to(device, non_blocking=True)
                vlogits = forward_logits(vinputs)
                vloss = criterion(vlogits, vlabels)
                running_vloss += vloss.item() * vinputs.size(0)
                n_val_seen += vinputs.size(0)

                if TEST_MODE and vbatch_idx >= 4:
                    print(f'  [TEST_MODE] обработано 5 батчей val, выходим')
                    break

        avg_vloss = running_vloss / n_val_seen
        print(f'  val_loss {avg_vloss:.4f}')

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            torch.save(model.state_dict(), best_path)
            print(f'  сохранены веса лучшей модели: {best_path}')

    if device.type == 'cuda':
        torch.cuda.synchronize()
    train_time = time.perf_counter() - train_start

    print(f'Обучение заняло {train_time:.1f} сек')
    return train_time


def evaluate_model(model_name):
    """Загружает лучшие веса, прогоняет тест. Возвращает (accuracy, inference_time_sec, n_test)."""
    print(f'\n#########\n Оценка: {model_name} \n#########\n')

    model, test_loader, test_size, forward_logits = _setup_model_for_eval(model_name)

    labels_predicted = []
    labels_true = []

    if device.type == 'cuda':
        torch.cuda.synchronize()
    inf_start = time.perf_counter()

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = forward_logits(images)
            _, predicted = torch.max(logits, 1)
            labels_predicted.extend(predicted.cpu().numpy())
            labels_true.extend(labels.cpu().numpy())

            if TEST_MODE and batch_idx >= 4:
                print(f'  [TEST_MODE] обработано 5 батчей test, выходим')
                break

    if device.type == 'cuda':
        torch.cuda.synchronize()
    inf_time = time.perf_counter() - inf_start

    n_seen = len(labels_true)
    acc = accuracy_score(labels_true, labels_predicted)

    # classification_report может ругаться, если в TEST_MODE не все классы попали в тестовые батчи
    try:
        report = classification_report(labels_true, labels_predicted,
                                       target_names=CLASS_NAMES, zero_division=0)
        print(report)
    except Exception as e:
        print(f'  classification_report не удался: {e}')

    print(f'Accuracy: {acc:.4f}, инференс: {inf_time:.2f} сек на {n_seen} картинок')
    # Возвращаем n_seen вместо test_size - в TEST_MODE это меньше
    return acc, inf_time, n_seen


# ----- список моделей -----
model_names = [
    # базовые
    "microsoft/swin-base-patch4-window7-224",
    "google/vit-base-patch16-224",
    "facebook/convnextv2-base-22k-224",
    "microsoft/cvt-21",
    "facebook/dinov2-base",
    "maxvit_base_tf_224.in1k", # timm
    # тяжёлые
    "microsoft/cvt-w24-384-22k",
    "microsoft/swin-large-patch4-window7-224",
    "google/vit-large-patch16-224",
    "facebook/convnextv2-large-22k-224",
    "facebook/dinov2-large",
    "maxvit_large_tf_224.in1k", # timm
]

# ----- прогон -----
results = []

for name in model_names:
    try:
        t_train = train_pipeline(name)
        acc, t_inf, n_test = evaluate_model(name)
        results.append({
            'model': name,
            'accuracy': acc,
            'train_time_sec': t_train,
            'inference_time_sec': t_inf,
            'inference_ms_per_image': 1000 * t_inf / n_test,
        })
    except Exception as e:
        print(f'!!! Модель {name} упала: {e}')
        import traceback
        traceback.print_exc()
        results.append({
            'model': name,
            'accuracy': None,
            'train_time_sec': None,
            'inference_time_sec': None,
            'inference_ms_per_image': None,
            'error': str(e),
        })

# ----- итоговая таблица -----
print('\n#########\n СВОДНАЯ ТАБЛИЦА \n#########\n')
if TEST_MODE:
    print("!!! TEST_MODE - результаты не имеют смысла, это проверка работоспособности\n")

header = f"{'model':<45} {'acc':>8} {'train, s':>10} {'infer, s':>10} {'ms/img':>10}"
print(header)
print('-' * len(header))
for r in results:
    if r['accuracy'] is None:
        print(f"{r['model']:<45} {'FAIL':>8}")
        continue
    print(f"{r['model']:<45} {r['accuracy']:>8.4f} {r['train_time_sec']:>10.1f} "
          f"{r['inference_time_sec']:>10.2f} {r['inference_ms_per_image']:>10.3f}")

# ----- CSV -----
fieldnames = ['model', 'accuracy', 'train_time_sec', 'inference_time_sec', 'inference_ms_per_image']
with open(results_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in results:
        writer.writerow({k: r.get(k) for k in fieldnames})
print(f'\nРезультаты сохранены в {results_csv}')