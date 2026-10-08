# Zotero MCP：分類、筆記與註解

0.3.0 新增九個工具，與文獻研究工具共用 AcademicHelper stdio server。key 僅由 `ZOTERO_API_KEY` 讀取，文庫由 `ZOTERO_LIBRARY_ID` 與 `ZOTERO_LIBRARY_TYPE`（user／group）決定。Windows 程序環境優先，缺少時讀取持久化 User 環境；明確空值不被覆蓋。所有工具每次讀目前環境，不載入明文設定檔。

## 研究者可以直接說

- 「把這批文獻依研究主題分類，先給我看預覽。」
- 「將我的 PDF 標記整理成閱讀筆記，保留頁碼與原文。」
- 「這篇能支持論文哪個論點？把用途與限制存成引用註釋。」
- 「將核對過 DOI 的文章加入這個文庫，保留已存在的書目。」

助手依 [zotero-library](../skills/zotero-library/SKILL.md) 選工具，帳號連接依 [setup-zetero](../.agents/skills/setup-zetero/SKILL.md)。分類建議由研究證據與目標決定，工具不替代正式納入判斷。

## 工具與操作順序

| 工具 | 能力 |
| --- | --- |
| `zotero_status` | 所選文庫唯讀存取；不測試寫入 |
| `zotero_search` | 書目搜尋，依集合／標籤／類型篩選，分頁 |
| `zotero_read_item` | 書目或附件及子項；讀筆記 HTML 與已同步 PDF 註解 |
| `zotero_list_collections` | 集合、父集合與版本，分頁 |
| `zotero_list_tags` | 既有標籤，分頁 |
| `zotero_create_collection` | 同名同父集合查重；預覽／建立 |
| `zotero_organize_items` | 保留既有集合與人工標籤；批次增補及移除 AH: 管理標籤 |
| `zotero_save_note` | AI 筆記／引用註釋預覽、建立、受保護更新 |
| `zotero_import_bibliography` | 有 DOI 的 journalArticle 查重及預覽／匯入 |

各列表 `limit` 為 1–100，`start` 為非負；依 has_more／next_start 續查，不把一頁當全庫。PDF 註解在附件 children 下，先從書目取得 attachment key，再讀該附件。工具讀取的是 API 中的註解與書目，不是 PDF bytes，沒有下載或全文解析。

寫入工具預設 `apply=false`。先讀預覽，保持研究者指定範圍；已明確授權實際整理時才使用 `apply=true`。分類套用必須提供預覽的 `expected_versions`；更新筆記需其 `expected_version`。HTTP 412 表示同步內容改變，不盲目重試。批次最多 50 筆，不是原子交易，逐筆保存成功／失敗；新 collection 或 bibliography POST 有穩定 write token，仍需在結果未知時先查當前資料。

典型分類預覽參數：

```json
{
  "item_keys": ["ABCD1234"],
  "add_tags": ["AH:閱讀:已讀"],
  "remove_managed_tags": ["AH:閱讀:待讀"],
  "add_collections": [],
  "apply": false
}
```

key 是示意，不能直接操作；讀取實際 key。套用時沿用相同變更，加上從預覽取得的 expected_versions。工具不移除集合歸屬、不刪書目、不改一般人工標籤。一般標籤可增補，狀態標籤以 AH: 作助手管理範圍；避免同義名稱與未經依據的品質標籤。

## 筆記保存與人工保護

筆記有 AI 說明、來源位置、穩定 note_id、助手標籤及 HTML 內的內容指紋。指紋不是研究品質或安全簽章，只用來偵測內容是否與上次工具產出一致。人工修改、舊版未帶指紋或 Zotero 正規化 HTML 都會保守阻擋覆寫，回傳 `human_modified_note`。助手保留原筆記，可另建新 ID 的合併筆記；不把指紋不一致當成內容不正確。

body／title／source_locator 接受純文字，由工具跳脫；引用文字必須來自實際閱讀的來源。工具不自動查核研究結論，助手須區分原文、使用者評論、AI 綜合及閱讀層級。已有相同內容重跑不建重複筆記；其他人工 child notes 保留。

## 本版邊界與驗證

目前不提供 PDF 註解建立／修改、附件下載／上傳、無 DOI 或非 journalArticle 的匯入。既有本機 zotero-literature-import 仍處理完整入庫與 PDF，需自行確認已安裝；不是 MCP 工具的內建依賴。

實作依 Zotero Web API v3 的分頁、items／collections、POST write results 與版本前置條件契約。官方參考：[基本請求](https://www.zotero.org/support/dev/web_api/v3/basics)、[讀取](https://www.zotero.org/support/dev/web_api/v3/read_requests)、[寫入](https://www.zotero.org/support/dev/web_api/v3/write_requests)、[檔案上傳](https://www.zotero.org/support/dev/web_api/v3/file_upload)。本次 Cloud 存取官方文件受網路限制，沒有真實 key；mock 契約測試與 stdio discovery 不代表官方 API 已實測，真實書目／分類／筆記寫入相容性仍需在有授權的文庫驗證。唯讀可先驗證，不能為安裝測試擅自寫入。

有界回應、固定官方 HTTPS host、禁止重新導向、已驗證 TLS、受控錯誤訊息與每次關閉 client 都由共用層處理。HTTP 錯誤不回傳私有 raw body；寫入網路失敗為 `write_outcome_unknown`，不是成功或確定未寫入。資料回傳本身仍是研究者的私有研究內容，不能自動外傳。成果按 [資料夾約定](output-layout.md) 整理。

維護者離線測試：`python -m pytest -q tests/test_zotero.py`。
