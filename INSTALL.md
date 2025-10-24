# 🧠 Audio2Text Whisper — инструкция по работе

---

## 📂 Куда класть аудио

После распаковки или клонирования создай (если нет) папку **audio** и помести туда записи:
```
audio/
├── meeting1.mp3
├── lecture.wav
└── voice_note.m4a
```
Форматы: `.mp3`, `.wav`, `.m4a`, `.mp4`.

---

## ⚙️ Что делает программа (по шагам)
1️⃣ Находит все файлы в `audio`.  
2️⃣ Очищает и нормализует звук.  
3️⃣ Распознаёт речь через модель Whisper.  
4️⃣ Сохраняет результат в `output` в виде `.txt`.

Пример:  
```
audio/interview.mp3 → output/interview.txt
```

---

## 📦 Где искать результаты
| ОС | Путь к папке с результатом |
|:--|:--|
| Windows | `C:\Audio2Text-Whisper-Ready-main\output\` |
| Linux | `~/Audio2Text-Whisper-Ready/output/` |

---

## 🪟 Windows 10 / 11 (из исходных файлов)

1️⃣ Скачать проект → зелёная кнопка **Code** → **Download ZIP**.  
2️⃣ Распаковать архив (`Audio2Text-Whisper-Ready-main.zip`).  
3️⃣ Установить Python с [python.org/downloads/windows](https://www.python.org/downloads/windows/) (с галочкой **Add to PATH**).  
4️⃣ Открыть распакованную папку → **Открыть в PowerShell**.  
5️⃣ Ввести:
   ```powershell
   py -3 -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   python main.py --batch
   ```
6️⃣ Подождать — после обработки файлы появятся в `output`.

---

## 🐧 Linux (Ubuntu)

```bash
sudo apt update
sudo apt install -y git ffmpeg python3-venv
git clone https://github.com/OmeNikArch/Audio2Text-Whisper-Ready.git
cd Audio2Text-Whisper-Ready
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p audio
python main.py --batch
```
Результаты → папка `output/`.

---

## ❓ Если что-то не работает
| Проблема | Решение |
|:--|:--|
| `ffmpeg not found` | `sudo apt install ffmpeg` (или `winget install Gyan.FFmpeg`) |
| Нет Python | Установить Python 3.x с официального сайта |
| Нет файлов в output | Проверить, что файлы в папке `audio` |
| Долго работает | Подождать — идёт распознавание ИИ |

---

## 🔄 Как обновить
Если проект скачан через Git:
```bash
cd ~/Audio2Text-Whisper-Ready
git pull
```
Если через ZIP — скачать новый архив и заменить папку.

---

## 💡 Советы
- Можно класть сразу несколько файлов — все обработаются.  
- Не переименовывай папки `audio` и `output`.  
- Первый запуск скачивает модель — ждать нормально.  
- Чтобы удалить программу — просто удали папку.

---

### 🆘 Нужна помощь?
Открой раздел **Issues** на странице GitHub и напиши, что именно не работает.
