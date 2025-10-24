# 🎧 Audio2Text Whisper

Простая программа для превращения аудио- и видеозаписей в текст (.txt) с помощью искусственного интеллекта **Whisper**.

---

## 🧩 Что делает программа
- Принимает аудио или видео-файлы (.mp3, .wav, .m4a, .mp4 и др.)
- Удаляет шум и улучшает звук
- Распознаёт речь и переводит её в текст
- Сохраняет результат в папку `output` в виде `.txt` или `.srt` (субтитры)

---

# 🚀 Как установить и запустить программу

---

## 🪟 **Для Windows 10 / 11**

### 1️⃣ Скачать программу с GitHub
1. Открой страницу проекта:  
   👉 **https://github.com/OmeNikArch/Audio2Text-Whisper-Ready**
2. Нажми зелёную кнопку **Code** → **Download ZIP**.  
   ![Скриншот Download ZIP](https://docs.github.com/assets/cb-42544/images/help/repository/code-button-download-zip.png)
3. В папке «Загрузки» найди `Audio2Text-Whisper-Ready-main.zip` и распакуй («Извлечь всё…»).

---

### 2️⃣ Установить Python (если не установлен)
1. Перейди на **https://www.python.org/downloads/windows/**  
2. Нажми **Download Python 3.x.x**, при установке поставь галочку **“Add Python to PATH”**.
3. После установки проверь:
   ```powershell
   py --version
   ```
   Если видишь что-то вроде `Python 3.12.3` — всё готово.

---

### 3️⃣ Подготовить и запустить
1. Открой распакованную папку `Audio2Text-Whisper-Ready-main`.  
2. Щёлкни правой кнопкой → **Открыть в PowerShell**.  
3. Введи команды по порядку:
   ```powershell
   py -3 -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   python main.py --batch
   ```
   
   
4. Первый запуск может занять несколько минут (скачивание модели Whisper).  
5. Готовые `.txt` файлы будут в папке `output`.

---

### 4️⃣ Куда класть аудио
Положи файлы в папку **audio** (создай её, если нет):
```
audio/
├── meeting1.mp3
├── interview.wav
└── note.m4a
```
Каждому файлу соответствует свой `.txt` в папке `output`.

---

## 🐧 **Для Linux (Ubuntu 20.04 / 22.04 / 24.04)**

### 1️⃣ Скачать и установить зависимости
```bash
sudo apt update
sudo apt install -y git ffmpeg python3-venv
git clone https://github.com/OmeNikArch/Audio2Text-Whisper-Ready.git
cd Audio2Text-Whisper-Ready
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt
```

---

### 2️⃣ Добавить аудио и запустить
```bash
mkdir -p audio
# скопируй свои аудиофайлы в audio/
python main.py --batch
```
После завершения → файлы в папке `output/`.

---

## 💡 Подсказки
- Первый запуск может быть долгим — идёт скачивание модели.  
- Можно класть несколько файлов — все обработаются.  
- Чтобы удалить программу, просто удали папку проекта.

---

### 📬 Обратная связь
Если что-то не работает — нажми **Issues** на странице GitHub и опиши, что именно произошло.
