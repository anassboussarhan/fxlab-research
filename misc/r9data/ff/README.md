---
license: mit
language:
- en
task_categories:
- time-series-forecasting
- tabular-classification
- tabular-regression
tags:
- forex
- economic-calendar
- financial-data
- macro-economics
- trading-strategy
- time-series
- market-events
- selenium
- pandas
- scraper
pretty_name: Forex Factory Economic Calendar (2007–2025)
size_categories:
- 10K<n<100K
---

# 📅 Forex Factory Economic Calendar Dataset (2007-01-01 to 2025-04-07)

This dataset contains a comprehensive archive of macroeconomic calendar events sourced from [Forex Factory](https://www.forexfactory.com/calendar), spanning from **January 1, 2007** to **April 7, 2025**.  
Each row captures a specific event with detailed metadata including currency, event type, market impact level, reported values, and descriptive context.

---

## 📦 Dataset Summary

- **Total timespan:** 2007-01-01 → 2025-04-07  
- **Format:** CSV (UTF-8)  
- **Timezone:** Asia/Tehran  
- **Detailed fields included**

---

## 🧾 Columns

| Column      | Description                                                                 |
|-------------|-----------------------------------------------------------------------------|
| `DateTime`  | Date and time of the event in ISO 8601 format                               |
| `Currency`  | Currency affected by the event (e.g., USD, EUR, GBP)                        |
| `Impact`    | Impact level: Low, Medium, or High Impact Expected                          |
| `Event`     | Name of the economic event                                                  |
| `Actual`    | Reported value of the indicator (if available)                              |
| `Forecast`  | Forecasted value                                                            |
| `Previous`  | Previously reported value                                                   |
| `Detail`    | Rich description including source, frequency, and why it matters to traders |

---

## 🔍 Example

```csv
2007-01-01T04:30:00+03:30,CNY,High Impact Expected,Manufacturing PMI,54.8,,55.3,"Source: CFLP (latest release) | Measures: Level of a diffusion index based on surveyed purchasing managers..."
```

---

## 💼 Use Cases

- Financial time series forecasting  
- Event-driven backtesting in algorithmic trading  
- Economic policy impact analysis  
- Macro indicators modeling for global trends  
- Educational purposes in finance and economics

---

## ⚙️ How It Was Built

The data was collected using a custom Python scraper using:

- [Selenium](https://pypi.org/project/selenium/) (via `undetected-chromedriver`)  
- [pandas](https://pandas.pydata.org/) for CSV export and manipulation  
- Incremental scraping and detailed event parsing  
- Timezone-aware timestamping

GitHub repository (scraper code): [🔗 Link to repository](https://github.com/ehsanrs2/forexfactory-scraper)

---

## 🛡 Disclaimer

This dataset is intended for **educational and research purposes only**.

Data was collected from [Forex Factory](https://www.forexfactory.com/) using publicly accessible web content.  
This project is **not affiliated, endorsed, or sponsored by Forex Factory** in any way.

Please respect [Forex Factory’s Terms of Use](https://www.forexfactory.com/notices). Do not use this dataset for commercial purposes or redistribution without checking the original source's policies.

---

## 📚 Citation

If you use this dataset in your work, please cite it as:

```
@misc{forexcalendar2025,
  title={Forex Factory Economic Calendar Dataset (2007–2025)},
  author={Ehsan},
  year={2025},
  howpublished={\url{https://huggingface.co/datasets/yourusername/forex-factory-calendar}},
  note={Accessed: YYYY-MM-DD}
}
```

---

## 🏷 Tags

`forex` `financial-data` `economic-calendar` `macro-economics`  
`time-series` `trading-strategy` `pandas` `selenium` `market-events`

```


