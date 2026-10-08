# 助手用：安裝、更新與驗證

先讀 [操作入口](README.md)。本指南由助手執行，不作為一般研究者的操作清單。

## 1. 確認 host 與來源

辨識 Windows／macOS／Linux／Cloud、AI client、可用 shell、Python、uv、Node、MCP 設定能力與 skill discovery。檢查既有 AcademicHelper、其他 skills、研究工作區與 `git status --short`；僅確認憑證變數的名稱及存在狀態，不列印值。

正式來源為 `https://github.com/TLOGBen/AcademicHelper`。首次安裝取得 main；更新先 fetch 並比對。記錄實際 `git rev-parse HEAD`，全部元件從該 commit 取得。髒 checkout 保留原樣，可在另一個乾淨目錄取得來源；Cloud 已隔離，優先使用現有 checkout，不為一般設定建立 worktree。不要猜測某個短 SHA 永遠是最新版，也不要覆蓋使用者尚未提交的檔案。

## 2. 選擇安裝方式

| Host | 安裝方法 |
| --- | --- |
| Claude Code plugin | 使用 host 支援的 marketplace 安裝，或以 `claude --plugin-dir <repo 的絕對路徑>` 載入 checkout；先確認 client 的實際 CLI／版本支援 |
| Codex repository skills | `.agents/skills/` 中兩個 skills 可由相應 host 探索；六個 `skills/` 不能假設會一併自動探索 |
| Codex 個人 skills | 用 host 的安裝能力，或把下列完整目錄安裝至其實際 User skill 根目錄（通常 `~/.codex/skills/`）；查實際路徑，不寫死使用者名稱 |
| 其他 AI client | 先確認 skill 與 MCP 相容方式。可先讀正式 skill 並依流程工作；沒有安裝能力時不能宣稱已完成安裝 |

需要安裝的八份正式來源：

| 來源目錄 | 安裝後名稱 |
| --- | --- |
| `.agents/skills/setup-zetero/` | `setup-zetero` |
| `.agents/skills/pico-literature-search/` | `pico-literature-search` |
| `skills/research/` | `research` |
| `skills/suggest-direction/` | `suggest-direction` |
| `skills/relevance/` | `relevance` |
| `skills/can-this-work/` | `can-this-work` |
| `skills/committee-review/` | `committee-review` |
| `skills/zotero-library/` | `zotero-library` |

安裝完整葉目錄，包括 `agents/`、`references/`、`assets/`、`scripts/`（存在時）。不要只複製 SKILL.md。已存在的同名 skill 先做差異比對；保留需要的本機修改，備份放在 active skill discovery 之外，以免載入重複版本。備份僅限非敏感 skill 檔案；憑證及個人設定不要帶入。逐檔比對來源與安裝結果，記錄保留的差異。

既有 `zotero-literature-import` 由本機流程另行維護，目前不在本 repository。不要刪除、覆蓋或宣稱已隨本 plugin 安裝，也不要另造一套 setup-zetero。

## 3. 準備 runtime 與 MCP

Python 最低 3.11，建議 3.12。PICO 與 Zotero Python helpers 使用 standard library，單獨使用不需要啟動 MCP；MCP 使用 `pyproject.toml` 的依賴。有可用 lock 時依其安裝；本 repository 忽略 `uv.lock`，未有 lock 時不得聲稱鎖定依賴。從 repo 根目錄執行：

```sh
uv sync --python 3.12 --group dev
uv run python -m pytest -q
```

MCP 是 stdio 服務，沒有網站或監聽 port，不能以瀏覽器開啟 `localhost` 驗證。Claude plugin 的 `.mcp.json` 使用 `${CLAUDE_PLUGIN_ROOT}`；其他 client 依其 schema 合併 server 設定，保留所有既有 servers，使用實際絕對路徑：

```json
{
  "mcpServers": {
    "academic-helper": {
      "command": "uv",
      "args": ["run", "--directory", "<repository absolute path>", "academic-helper"]
    }
  }
}
```

此為設定片段，不能直接以 placeholder 使用或覆蓋整個設定檔。MCP client 應完成 initialize、列出十五個工具（六個研究工具與九個 Zotero 工具）並呼叫 `expand_topics`；輸入參數以當次 schema 為準。僅成功啟動程序不代表 handshake 成功。有 client 重載／重啟需求時，先完成其他步驟，再告訴研究者最少必要操作。

Excel 另需 Node 與相容的 `@oai/artifact-tool` 文件 runtime；Python 安裝不提供此套件。探索實際 runtime，必要時設定 `CODEX_NODE` 和 `CODEX_NODE_MODULES`。缺少時保留 HTML／CSV／JSON，使用實際可用的試算表工具完成 XLSX；未完成就標示，不偽造副檔名或宣稱完整交付。

## 4. 依需求準備網路與授權

| 能力 | 需要的網域／變數 |
| --- | --- |
| MCP 跨來源搜尋 | `api.openalex.org`、`api.semanticscholar.org`、`api.crossref.org`；`SEMANTIC_SCHOLAR_API_KEY` 選用 |
| PICO PubMed／PMC | `eutils.ncbi.nlm.nih.gov`、`pmc-oa-opendata.s3.amazonaws.com`；NCBI_EMAIL／NCBI_API_KEY 選用 |
| PICO 引用查核 | `api.openalex.org`；OPENALEX_API_KEY 選用 |
| Zotero | `api.zotero.org`；ZOTERO_API_KEY、ZOTERO_LIBRARY_ID、ZOTERO_LIBRARY_TYPE |

只增補必要網域，不替換未知的既有規則。Cloud 保存 draft 不等於套用或發布；當次實際連線、設定儲存與新環境驗證分別記錄。本機 Windows User 環境不會同步至 Cloud。

Zotero 依 [正式 skill](../.agents/skills/setup-zetero/SKILL.md) 與 [執行指南](../.agents/skills/setup-zetero/references/execution.md) 操作：先沿用既有環境，缺個人文庫 ID 才查候選，不把 discovery 當連接成功。群組尊重已選目標。key 只能從 ZOTERO_API_KEY 取得；安全授權介面不可用時，完成其他工作並說明授權仍待處理，不改存明文。安裝驗證只做唯讀 GET，不建立測試書目或上傳 PDF。

## 5. 驗證與交接

依實際安裝範圍選擇必要驗證；只更新文件／skills 不必反覆跑全部應用測試。

```sh
python -B -m unittest discover -s .agents/skills/setup-zetero/scripts -p test_zotero_doctor.py -v
python -B -m unittest discover -s .agents/skills/pico-literature-search/scripts -p test_literature_tool.py -v
python .agents/skills/pico-literature-search/scripts/literature_tool.py plan --out outputs/install_plan_check
```

最後一條只驗證內建 GSEOH 範例的離線策略輸出，不能當作研究者題目已搜尋。使用新的輸出位置避免覆蓋；真實搜尋與 PDF 依需求做有界檢查，詳見 [Cloud 指南](../docs/codex-cloud-literature.md)。確認 HTML、表格列數、PDF 實際可讀及來源相符。記錄來源 commit、安裝目錄、檔案比對、通過／失敗／未跑的檢查、可用工具、未完成能力與重啟需求，再轉入 [研究專案初始化](project-init.md)。
