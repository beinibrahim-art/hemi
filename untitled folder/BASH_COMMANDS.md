# أمر التشغيل السريع 🚀

## الطريقة الأسرع (سطر واحد):

```bash
bash quick_start.sh
```

أو:

```bash
./quick_start.sh
```

---

## أمر مباشر (بدون ملف):

```bash
cd "untitled folder" && python3 -m venv venv 2>/dev/null; source venv/bin/activate; pip install -q -r requirements.txt 2>/dev/null; python3 app.py
```

---

## أمر مبسط (3 أسطر):

```bash
cd "untitled folder"
source venv/bin/activate 2>/dev/null || (python3 -m venv venv && source venv/bin/activate && pip install -q -r requirements.txt)
python3 app.py
```

---

**بعد التشغيل:** افتح http://localhost:5000
