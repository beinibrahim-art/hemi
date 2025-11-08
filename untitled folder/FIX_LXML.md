# حل مشكلة lxml على macOS

## المشكلة
فشل تثبيت `lxml` بسبب مشكلة في الترجمة (compilation error)

## الحلول

### ✅ الحل 1: تثبيت lxml من binary wheel (موصى به)
```bash
source venv/bin/activate
pip install --only-binary lxml lxml
```

### ✅ الحل 2: تثبيت libxml2 أولاً
```bash
# على macOS
brew install libxml2 libxslt

# ثم ثبت lxml
pip install lxml
```

### ✅ الحل 3: استخدام html.parser بدلاً من lxml (لا يحتاج تثبيت)
الكود يعمل بدون lxml! BeautifulSoup4 يمكنه استخدام `html.parser` بدلاً من `lxml`.

**لا حاجة لتثبيت lxml** - التطبيق يعمل بدونها!

### ✅ الحل 4: تثبيت lxml من source مع إعدادات خاصة
```bash
source venv/bin/activate
export CFLAGS="-I$(brew --prefix libxml2)/include/libxml2"
pip install lxml
```

## التوصية

**لا تقلق!** lxml اختياري. التطبيق يعمل بدونها باستخدام `html.parser` المدمج في Python.

إذا أردت تثبيتها لاحقاً:
```bash
brew install libxml2 libxslt
pip install lxml
```
