# 無 PDF 文獻的優先取得與 COSMIN

使用者指定：PDF 取不到時保留標題，優先引用較多及研究品質較高的文獻，整理固定格式；「論文分數」明確指研究品質（例如 COSMIN）。

## 品質判讀

依 [COSMIN V2 手冊](https://www.cosmin.nl/wp-content/uploads/COSMIN-manual-V2_final.pdf)及當次適用工具，RoB 對各測量特性評方法品質，通常為 very good／adequate／doubtful／inadequate，以 box 內 worst score counts 作該特性評級；不是把整篇文章的分數加總或平均。測量特性結果的 sufficient／insufficient／indeterminate 與研究方法品質亦分開；研究群體的證據確定性與單篇評級分開。

全文未取得（也無完整 HTML 或經核實系統性回顧所提供的對應評估）不能憑標題、摘要、Cronbach alpha 或引用數產生 COSMIN 高品質判定。標「尚未取得全文，COSMIN 待評」。若有外部已發表 COSMIN 評估，核對 DOI、版本、樣本、評估屬性，列來源及表格位置並標「外部評估」；不能把另一量表／另一語言版本的評分套入。

完整 HTML 可以支持評估，無 PDF 不等於無全文；對未報告標準依該 COSMIN 工具的規則評，不把缺資料當很好。每個評級附頁／表／段證據、工具版本及評估日期。原始開發、翻譯、驗證及單純使用研究區分，不把單純使用當所有特性有驗證。

評級格式可存為 record.quality_assessments：

```json
[{"property":"structural validity", "rating":"adequate", "tool":"COSMIN 指定版本", "basis":"全文或外部評估", "source":"實際網址或檔案", "location":"頁／表／段", "assessed_at":"實際日期"}]
```

這只是資料 schema，例值不代表任何本次文章實際獲評。不同屬性不得無規則混成單篇總分；若選「結構效度」排序，未評該屬性者另列待評。臨床介入研究另用設計適合的工具（例如 RoB 2／ROBINS-I），不強制 COSMIN。

## 引用數

預設用 OpenAlex 的 article `cited_by_count`，按 DOI 精確匹配。官方 [API 使用示例](https://help.openalex.org/how-to/api-recipes/)、[驗證及 key](https://help.openalex.org/api/authentication/)。2026-10 查核當下可免 key 作基本查詢；需要較高用量時用使用者 OPENALEX_API_KEY，不代註冊或購買。批次最多 50 DOI 一輪保守使用，來源、日期、OpenAlex work URL 都保留。

OpenAlex 与 Scholar／Scopus 引用數不同，不混成同一比較欄。來源未索引或請求失敗、沒有 DOI 或匹配未確定均留空／待查，只有來源明確回 0 才寫數值 0。無 DOI 时可人工題名 + 年 + 作者核實，不用模糊第一筆自動冒充精確匹配。

## 優先表與排序

| 順位 | 文章標題 | 年份 | DOI／PMID | 文獻範圍 | 引用數（來源／日期） | COSMIN 各特性評級／證據 | PDF 狀態／取得入口 |
|---|---|---|---|---|---|---|---|

先保留原量表、已知翻譯和直接對應研究，不能因引用少淘汰。可比較品質評級時按同特性品質優先、同級引用數降序；未知品質另列待評区，依引用數作取得排序，不能宣稱是品質最高清單。不同年份受引用累積時間影響，不把高引用當證據品質。

Helper `rank --config <profile> --out <existing-output> --top 20` 只自動補引用數、保留既有 quality_assessments 並生成待取得清單；不自動給 COSMIN 分數。代理有可靠全文時完成屬性評估並整理表，無資料時保留待評。Excel 新增待取得全文頁、JSON 保存完整欄位與來源。清楚說明所列排序依據和未知資訊。
