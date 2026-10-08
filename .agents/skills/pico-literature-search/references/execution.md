# Helper、輸入與可驗證產出

## 位置和依賴

Python helper 只用 standard library。Excel 用 `export_excel.mjs` 與 bundled artifact-tool；`load_workspace_dependencies` 找 Python、Node、packages。本機 resolver 有 primary runtime 預設；其它 host 可設 `CODEX_NODE` 實際 Node、`CODEX_NODE_MODULES` node_modules 或父目錄，不改 runtime。

在研究工作區執行，或設 `LITERATURE_WORKSPACE`。`local_pdf` 相對於該工作區，亦可絕對路徑。`--out` 指工作成果目錄，不用安裝的 skill。run 遇既有 manifest 停止以保筆記；import 更新既有目錄。

## 新題目 profile

代理依研究寫 UTF-8 JSON：

```json
{
  "topic": "研究題目",
  "pico": {"P":"族群", "I":"介入或量表", "C":"可不適用", "O":"結局或測量特性"},
  "queries": [
    {"id":"broad", "database":"PubMed / NCBI E-utilities", "label":"主搜尋", "query":"已研究的實際 PubMed 搜尋式"},
    {"id":"cochrane", "database":"Cochrane Library", "label":"Search Manager", "query":"已研究的逐行搜尋式"},
    {"id":"scholar1", "database":"Google Scholar", "label":"原題名", "query":"短搜尋式"}
  ],
  "supplemental_records": []
}
```

可加 `seed_pmid` 為核實的原文 PMID；自訂 queries 不回退 GSEOH。id 唯一，PubMed database 以 PubMed 開頭。只把 PubMed IDs 傳 `run --queries`，helper 不用 API 執行 Scholar／Cochrane。前三種資料庫 url 可省，其它來源提供 url。

補入資料可含 title、数值 year、authors、journal、doi、pmid、pmcid、url、kind、source、local_pdf，標人工補入／使用者附件／既有檔，不捏造識別碼。原始論文、問卷和研究報告各自標明。

GSEOH profile 的英文詞彙生成器產生 names、methods_broad、methods_population、translations、community、locale 六輪及 Cochrane／Scholar 詞。換量表時用自訂 queries。

## 參數和匯入

plan 無網路請求，產 search_plan.json、搜尋式.html。run 每輪預設 500 筆、最多嘗試 20 篇 PDF；依範圍調整上限。import 讀 UTF-8 RIS：Scholar 選 RefMan，Cochrane RIS，填來源及查詢 ID。RIS 缺 ER 停止不漏筆，其它編碼先保留原檔轉 UTF-8。

`NCBI_EMAIL`、`NCBI_API_KEY` 可選；每請求間隔至少約 .42 秒，有限重試。網路錯誤、取回不一致、API 警告及實際翻譯保留，不改零命中。

## 2026 PMC Cloud PDF

[OA service 停用](https://pmc.ncbi.nlm.nih.gov/tools/oa-service/)、[Cloud](https://pmc.ncbi.nlm.nih.gov/tools/cloud/)、[AWS API](https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/)、[README](https://pmc-oa-opendata.s3.amazonaws.com/README.txt)。

PMCID 用 ListObjectsV2 prefix `PMC<id>.` delimiter `/` 找 article-version，讀 `<prefix>/<prefix>.json`，使用 pdf_url。不假定 `.1` 必存在；高版號是處理次序，未必較新／較優。helper 優先非 manuscript，保留授權及版本。不能下載不等於網頁沒全文；非 CC manuscript 可能只有 XML／文字。

接受官方 bucket metadata 的 PDF URL，限 60 MB，檢查 PDF magic、URL 如附的 MD5，另記 SHA256；再開檔核對文獻與可讀性。raw 保存 API 與 metadata，不保存 key。

## 閱讀與匯出

manifest.json 含完整摘要；文獻清單.csv 是 UTF-8 BOM；文獻清單.xlsx 有文獻清單／搜尋紀錄，執行 rank 後另加待取得全文頁；另有搜尋式.html、pdfs/。PMID／DOI／PMCID 為文字，年為數值，摘要前 240 字標節錄，最後三欄供篩選和筆記。

`rank --config <profile.json> --out <existing-output-dir> --top 20` 批次以 DOI 查 OpenAlex 引用次數，輸出待取得全文 JSON 及 Excel 的待取得全文頁；不自動評 COSMIN。引用 API 可選 OPENALEX_API_KEY，不保存 key。新輸出 XLSX 檔名為 `文獻清單.xlsx`。

去重先 PMID 再 DOI，無共同識別碼才看一致題名、年份。不同識別碼／年份或缺資料保留人工判斷。來源和查詢可回溯，不刪同研究不同報告。

Excel 手動筆記不反向同步 JSON。匯入前先保留有筆記的 Excel，或用 PMID／DOI 建獨立筆記檔合併。summary.json 是初始 run 摘要，import 後用 manifest／Excel，不拿舊摘要報新數。

新建 XLSX 套用 spreadsheets marker／render／verify。Helper recalculate、inspect、render、export；查看所有工作表預覽及下載 PDF，重新開啟已匯出的 XLSX 確認表格名稱唯一。測試：`<python> -m unittest discover -s <skill>/scripts -p test_literature_tool.py -v`。測試驗證日期／IDs／摘要、RIS 完整性、去重來源、PDF 型別及新題目不套 GSEOH，不能代替真實資料庫驗證。測試暫存目錄設在可寫工作區，不寫入安裝的 skill。
