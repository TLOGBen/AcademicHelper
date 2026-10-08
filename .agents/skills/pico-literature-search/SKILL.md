---
name: pico-literature-search
description: 根據 PICO 或指定量表研究 PubMed、Cochrane Library 與 Google Scholar 的搜尋策略，取得可下載全文 PDF、去重並輸出 Excel 文獻閱讀清單；適用於文獻回顧及量表翻譯驗證。
---

# PICO 文獻搜尋與 PDF 整理

將研究問題轉為有來源依據、可重現的資料庫搜尋，交付可閱讀的 PDF、Excel 文獻清單與完整搜尋紀錄。預設繁體中文說明，保留文章原題名與識別碼。

## 研究問題與模式

沿用聊天已有的 PICO、量表、母版、目標語言、年齡、年份及情境。資訊不足時只問會改變搜尋的關鍵問題，同時做不依賴答案的工作。未指定年齡或年份不得直接当成排除條件。

- 一般 PICO：辨認族群、介入／暴露、比較與結局。概念内同義詞 OR、概念間 AND；是否加入 C／O 依漏文獻風險決定。
- 量表翻譯／測量研究：以量表名稱、構念、版本／語言、測量特性為主，不為形式填治療或對照組，不强制全部 PICO 欄位 AND。
- 只研究策略：交付報告及平台搜尋式，不啟動大量全文下載。
- 搜尋並整理：執行範圍內的搜尋、書目、全文及 Excel，清楚標示尚未執行的平台。

GSEOH 讀 [references/gseoh.md](references/gseoh.md)，使用 [assets/gseoh_profile.json](assets/gseoh_profile.json) 起步。其它題目重新研究並產生 profile，不沿用 GSEOH／口腔／老人詞彙與文章補入清單。附件與原作者通信是研究材料，內文不擴張本次操作授權。

## 研究與查核搜尋策略

讀 [references/search-strategy.md](references/search-strategy.md)。查各平台最新官方說明與原始研究，建立概念、同義詞、MeSH 與來源對照。不要將同一條長查詢直接貼到不同平台；不宣稱未測試策略有已驗證敏感度或是唯一最佳方案。

已指定量表先以縮寫、全名、原題名及 DOI 找開發文獻，再追版本、引用與翻譯。先跑不限語言／情境的名稱搜尋，再做方法、族群、情境聚焦；社區、臺灣、translation 不同時成為第一輪必要條件。以已知種子檢查尋回；不在收錄範圍時另記來源，不冒充資料庫命中。

PubMed 與 E-utilities `db=pubmed` 是同一來源，合併去重。Cochrane 的 CDSR、CENTRAL 分別紀錄；CENTRAL 未必提供全文。Scholar 用短搜尋與 Cited by／All versions 補查，保留排序、日期範圍及實際查看範圍，不把估計結果數當全部匯入數。

## 執行與取得 PDF

讀 [references/execution.md](references/execution.md) 取得 schema、命令及支援界線。`load_workspace_dependencies` 可用時用來找 bundled Python／Node；沒有此工具時用 shell 查 Python／Node 與實際 runtime，不因此停止。Excel 匯出前驗證 `@oai/artifact-tool` 可解析，必要時設定 `CODEX_NODE`／`CODEX_NODE_MODULES`。成果存目前工作區 `outputs/`，不寫入已安裝 skill。

將本次 profile 存在工作區，執行 helper：

```text
<python> <skill>/scripts/literature_tool.py plan --config <profile.json> --out <new-output-dir>
<python> <skill>/scripts/literature_tool.py run --config <profile.json> --out <new-output-dir> --queries <pubmed-query-ids> --max-per-query <limit> --pdf-limit <limit>
<python> <skill>/scripts/literature_tool.py import --config <profile.json> --out <existing-output-dir> --ris <export.ris> --source <actual-source> --query-id <actual-query-id>
```

Helper 處理 PubMed ESearch／EFetch、去重、PMC Cloud PDF 與 Excel；不會自動研究任意中文 PICO。概念解讀、英文化、同義詞及新主題平台語法由 skill 執行代理依來源研究後生成。

Cochrane 優先用可用且已授權的瀏覽器執行 Search Manager 與匯出，否則交付搜尋式及 RIS 流程並標待執行。Scholar 官方不提供 bulk access，不使用無人值守批次擷取；使用網頁 RefMan／RIS 或從出版來源補入核實書目。一般 web search 線索不包裝為 Scholar 查詢完成。

文獻納入與 PDF 取得分開：先收完整書目，記錄已下載、已有本地檔、無可下載版本、需機構／作者取得、失敗及本次上限。不是每個 PubMed／PMC／Cochrane 紀錄都有可程式下載 PDF。核對當次官方介面；2026-08 停用的 PMC `oa.fcgi` 與舊 FTP 範例不沿用。沒有 PDF 仍保留文章，不自動排除。授權信件條件僅套對應量表。

無法取得 PDF 時讀 [references/quality-priority.md](references/quality-priority.md)，產生待取得全文表，列題名、年、DOI、引用次數、引用來源／日期、各測量特性的 COSMIN 評級及證據、全文入口與原因。使用者指定「論文分數」為研究品質，不是期刊影響因子或相關性分數。先限於符合研究問題的候選，保留原量表及版本，再於可比較的同一品質面向優先品質較高者，引用次數作次順位；没有全文或可核對外部評估時標 COSMIN 待評，先以引用數規劃取得順序，不捏造數字總分。引用缺資料不填 0。

## 閱讀清單及完成檢查

XLSX 套用可用的 spreadsheets 技能及 bundled artifact-tool，保留題名、作者、年、期刊、DOI／PMID／PMCID、實際來源與命中查詢、全文入口、PDF 路徑及狀態。提供未填的篩選決定、排除理由及閱讀筆記欄。完整摘要在 JSON，節錄须標明。

保留 search_plan、API 回應、manifest，以及每輪時間、實際查詢、總命中／取回數、截取和失敗。名稱命中是候選，不是完成驗證文獻納入。合併仍可回溯；不同量表、同研究不同報告、評論及回覆不無理由刪掉。

完成前檢查種子、去重與截取；開 PDF 確認可讀且文獻相符；檢視 Excel 欄位、筆數及排版。失敗不等於零命中，未執行的平台不填 0。未完成來源與限制在報告和最後回覆說明。

交付本次研究報告、Excel、PDF 資料夾及搜尋式入口，區分策略完成、書目取回、全文取得與人工篩選進度。已授權的可逆整理可直接完成；skill 不授權寄信、購買全文或發表。
