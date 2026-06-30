# Mean Reversion + ADX Bot — Setup Guide

## الخطوات

### 1. تثبيت المكتبات
```bash
pip install -r requirements.txt
```

### 2. إنشاء Demo Account على Bybit
1. روح: https://testnet.bybit.com
2. سجّل account جديد
3. روح: Account → API Management
4. اعمل API Key جديد وخلّيه Unified Trading
5. انسخ الـ API Key والـ Secret

### 3. تعديل الـ API Keys في bot.py
```python
API_KEY    = "your_key_here"
API_SECRET = "your_secret_here"
DEMO_MODE  = True   # خليه True للـ demo
```

### 4. تشغيل الـ Bot
```bash
# تشغيل عادي
python bot.py

# عرض الإحصائيات
python bot.py stats
```

---

## ملاحظات مهمة

**Limit Orders فقط:**
الـ bot بيستخدم Limit Orders مش Market Orders
عشان يدفع 0.02% (Maker) بدل 0.05% (Taker)

**Circuit Breaker:**
لو خسر 5% في آخر 10 صفقات → يوقف أسبوع تلقائياً

**Position Size:**
3.4% من الـ portfolio لكل صفقة (25% Kelly)

**مدة الصفقة:**
8 ساعات ثم إغلاق تلقائي

---

## على الـ Cloud (بعد الـ testing)

```bash
# DigitalOcean / VPS
nohup python bot.py > output.log 2>&1 &

# أو استخدم screen
screen -S bot
python bot.py
# Ctrl+A then D للـ detach
```

---

## الملفات
- `bot.py` — الـ bot الرئيسي
- `bot_log.txt` — logs مفصّلة
- `trades.json` — سجل الصفقات والـ state
