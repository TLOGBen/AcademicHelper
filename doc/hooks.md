# 研究進度與成果入口 hooks

0.3.1 提供兩個 command hooks，共用 standard-library Python 程式；沒有外部請求、Zotero 寫入、憑證載入或 transcript 讀取。一般研究者不需設定 hook 或交付紀錄，由助手安裝並完成下列流程。

| 事件 | 行為 |
| --- | --- |
| SessionStart | 有研究入口／研究概況時，提供檔案位置供助手接續；不輸出檔案內容，不改檔 |
| PostToolUse | 檔案／shell 操作後，讀取明確交付紀錄，更新研究入口與成果索引；可識別的失敗事件略過 |

## 兩方載入與相容條件

Claude Code 使用 `.claude-plugin/plugin.json` 與預設 `hooks/hooks.json`；`${CLAUDE_PLUGIN_ROOT}` 用在有引號的命令路徑，Windows 依 Claude 的 Git Bash 行為處理。Codex 使用 `.codex-plugin/plugin.json` 明確載入 `hooks/codex-hooks.json`，使用 `${PLUGIN_ROOT}`，並另提供 Windows cmd 的 `%PLUGIN_ROOT%` 命令。兩方均接收 stdin JSON，以 `hookSpecificOutput.hookEventName/additionalContext` 提供上下文，正常情況只輸出一個 JSON object。

Codex 來源核對 commit `14c8b7771ab2b617a131f5d8e55e98d18e56ed09`，Claude 官方範例 `71cdddec623889d38af14b7a489670a03186f659`，Anthropic SDK types `f7b0b62c2a8d110d4da0eec0aa70cf795ec3afc4`（2026-10-08 核對）。本次 Cloud CLI `0.159.0-alpha.3` 的 hooks feature 為 stable／true；不能據此認定其他桌面版本支援或已載入本 plugin。安裝助手須查當次版本、plugin 載入結果及 hook 信任狀態，不使用 bypass-hook-trust、不自行偽造 trusted hash。需要 host 的信任介面時，完成其餘工作後只請研究者做該必要操作。

Hook 命令以已安裝的 uv 執行可用 Python，使用 `--no-project --no-python-downloads`，不為觸發 hook 同步依賴、下載 Python 或寫 lockfile。先由安裝流程準備 uv 與 Python 3.11+。沒有 runtime、host 未支援、只安裝 User skills 或 hooks 被停用時，助手仍可直接執行下列 helper；不能稱為自動事件觸發。

## 助手交付流程

1. 沿用研究工作區，把成果放在新 `outputs/成果/<任務>/`。實際開啟並核對主要成果、附屬檔案與完成範圍。
2. 登記 `delivery.json`：用途、時間、主要檔案、附屬檔案、ready／partial／failed 與重要限制。沒有真實搜尋或全文時不能標成完整研究；工具只檢查檔案及紀錄，不驗證科學結論。
3. 更新固定入口；下列 `record` 命令會立即更新，即使沒有 hooks 也能完成交付。後續檔案操作觸發 PostToolUse 時會再核對，不重複新增區塊。
4. 回覆先提供研究入口與本次主要成果；根目錄人工題目、判斷與下一步由助手合併維護。

以下由助手換成實際路徑與任務，不交給研究者執行：

```text
<python> <plugin_root>/scripts/research_hooks.py record
  --workspace <研究工作區絕對路徑>
  --task <outputs/成果/內的單一任務資料夾名稱>
  --purpose 文獻搜尋
  --status partial
  --primary 開始閱讀.html
  --artifact 文獻清單.csv
  --limitation 全文與其他資料庫仍待補
```

這是參數展示，助手組成一條正確命令／argv；安全引用所有路徑與文字。`--artifact`／`--limitation` 可重複；只有策略時 primary 用 `搜尋式.html`，status 與 limitation 明確說明範圍。`failed` 可不提供 primary；可用成果與列出的附屬檔案需存在且非空。若只存在已安裝的葉 skill 而沒有 helper，可依 [schema](../hooks/delivery.schema.json) 寫紀錄，再由助手按 [資料夾約定](output-layout.md) 更新入口。

手動補更新：`<python> <plugin_root>/scripts/research_hooks.py refresh --workspace <研究工作區絕對路徑>`。時間保存 ISO 帶時區格式，入口顯示預設 Asia/Taipei，可由助手依研究者時區設定非敏感 `ACADEMIC_HELPER_TIMEZONE`。

## 保留資料與失敗處理

入口只修改 `academic-helper:deliveries:start/end` 之間的區塊，其他人工內容原樣保留；未有區塊則附加，不覆寫整份文件。標記不完整、符號連結超出工作區、檔案不存在或紀錄無效時保留原狀／略過該紀錄。不要手動編輯管理區塊；人工研究筆記放在區塊外。

每種用途選最新可用 ready／partial 成果，顯示限制；最新失敗或主要檔案失效時保留前次可用成果，旁註失敗，歷史仍可查。原始檔案不搬移、不刪除。入口以相對且 URI 編碼連結處理中文、空格與括號。這不驗證 HTML 內部連結、XLSX 內容或 PDF 題名，仍由助手交付前核對。

最多掃描約 500 個任務，單份紀錄最多 64 KiB；過量不做部分索引冒充完整。並行更新使用非等待的檔案鎖與原子檔案替換；鎖忙時保留原狀，下次再試，超過五分鐘的殘留鎖可回收。兩個入口分別原子更新，不是跨檔交易；中途失敗可重跑修復。Hook 錯誤是非阻擋回報，不將原始輸入、私有內容或 exception 印出，也不自動重跑遠端寫入。

## 驗證範圍

離線檔案測試涵蓋人工段落、重跑、最新失敗、部分完成、失效檔案、路徑與 symlink、鎖定、事件輸入及安全錯誤。另以官方 Codex output schemas 驗證 JSON 格式，並實際執行兩份 Unix command。Windows 命令僅能在本 Linux 環境核對格式；尚未在 Windows／Claude client 中實測，也未宣稱目前聊天會自動觸發新 hook。助手安裝後應用 host 的 hooks 列表／事件結果確認真正載入，再回報。

官方來源：[Codex hook 設定](https://github.com/openai/codex/blob/14c8b7771ab2b617a131f5d8e55e98d18e56ed09/codex-rs/config/src/hook_config.rs)、[Codex discovery](https://github.com/openai/codex/blob/14c8b7771ab2b617a131f5d8e55e98d18e56ed09/codex-rs/hooks/src/engine/discovery.rs)、[Claude plugin 範例](https://github.com/anthropics/claude-code/blob/71cdddec623889d38af14b7a489670a03186f659/plugins/explanatory-output-style/hooks/hooks.json)、[Anthropic SDK types](https://github.com/anthropics/claude-agent-sdk-python/blob/f7b0b62c2a8d110d4da0eec0aa70cf795ec3afc4/src/claude_agent_sdk/types.py)。
