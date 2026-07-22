# Ping Forecast — live demo 🌊

พยากรณ์อัตราการไหลของน้ำ (discharge) ที่สถานี **P.1 สะพานนวรัฐ เชียงใหม่**
ล่วงหน้า 1–7 วัน อัปเดตอัตโนมัติทุกวันด้วย GitHub Actions

**หน้า dashboard:** https://tittiyanatcheeze.github.io/ping-forecast-live/

เป็น deploy สาธารณะของแอพจากทีสิส *Hydrology-Informed Machine Learning for
Multi-Day Streamflow Forecasting in the Ping River Basin* — โมเดลถูกเทรนไว้แล้ว
repo นี้ทำแค่ inference + เก็บความแม่นแบบ prospective (ทำนายก่อน เฉลยทีหลัง)

## ทำงานยังไง

ทุกวัน 08:00 น. (ICT) GitHub Actions จะ:
1. ดึงน้ำท่ารายวัน 14 วันล่าสุดของ P.1/P.67/P.20/P.75 จาก endpoint RID ภาค 1
2. ดึงพยากรณ์ฝน 7 วันหน้าที่พิกัด P.1 จาก Open-Meteo
3. สร้าง feature set_1b (lag 1–7 ทั้ง 4 สถานี) แล้วพยากรณ์ 3 โมเดล:
   - **persistence** — Q(t+L) = Q(t) เส้นฐาน
   - **xgb_1b** — XGBoost ใช้น้ำท่า 4 สถานี
   - **xgb_1b_nwp** — เพิ่มฝนสะสมพยากรณ์ (แบบ Track B ของทีสิส)
4. บันทึกคำพยากรณ์ลง `forecast_log.csv`, เฉลยแถวที่ครบกำหนดกับค่าจริง,
   คำนวณ MAE/bias สะสม แล้ว render `docs/index.html`
5. commit ผลกลับ repo → `forecast_log.csv` ใน git history = หลักฐาน prospective
   ที่ timestamp แก้ย้อนหลังไม่ได้

## หมายเหตุสำคัญ

- ค่าจริงจาก endpoint RID เป็นค่า **provisional** (ต่างจากชุดข้อมูลทีสิส
  ~2–3 m³/s ที่ P.1) ใช้เพื่อ demo/verify เท่านั้น **ไม่นำไป retrain**
- โมเดลเป็นรายวัน horizon 1–7 วัน · เกณฑ์น้ำท่วม 416 m³/s (ตลิ่ง 3.70 ม.)
- ฝน NWP เป็นจุดกริดเดียวที่พิกัด P.1

## รันเอง

```bash
pip install -r requirements.txt
python forecast.py      # เขียน docs/index.html + forecast_log.csv
```
