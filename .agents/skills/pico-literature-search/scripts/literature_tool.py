"""GSEOH strategy, PubMed retrieval, RIS import, PMC Cloud PDFs and Excel export.

Python standard library only. Excel export uses Codex's bundled artifact-tool.
No Scholar/Cochrane scraping; their searches are generated for browser use.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(os.environ.get("LITERATURE_WORKSPACE", os.getcwd())).resolve()
HERE = Path(__file__).resolve().parent
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
CLOUD = "https://pmc-oa-opendata.s3.amazonaws.com/"
TAIPEI = timezone(timedelta(hours=8))


def stamp():
    return datetime.now(TAIPEI).isoformat(timespec="seconds")


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def field_terms(terms):
    if not terms:
        raise ValueError("關鍵詞區塊不可為空")
    clean = []
    for term in terms:
        if not isinstance(term, str) or not term.strip() or re.search(r'["\[\]()]', term):
            raise ValueError(f"請在設定檔輸入單純詞彙，不要輸入搜尋運算式：{term!r}")
        clean.append(f'"{term.strip()}"[tiab]' if " " in term else f"{term.strip()}[tiab]")
    return "(" + " OR ".join(clean) + ")"


def make_plan(cfg):
    if cfg.get("queries"):
        plan = []
        seen = set()
        for item in cfg["queries"]:
            if not all(item.get(k) for k in ("id", "database", "label", "query")) or item["id"] in seen:
                raise ValueError("自訂 queries 必須有唯一 id、database、label、query")
            seen.add(item["id"])
            item = dict(item)
            if item["database"].startswith("PubMed"):
                item.setdefault("url", "https://pubmed.ncbi.nlm.nih.gov/?" + urllib.parse.urlencode({"term": item["query"]}))
                item.setdefault("status", "尚未執行")
            elif item["database"] == "Google Scholar":
                item.setdefault("url", "https://scholar.google.com/scholar?" + urllib.parse.urlencode({"q": item["query"]}))
                item.setdefault("status", "待網頁搜尋／RIS 匯入")
            elif item["database"].startswith("Cochrane"):
                item.setdefault("url", "https://www.cochranelibrary.com/advanced-search/search-manager")
                item.setdefault("status", "待網頁搜尋／RIS 匯入")
            elif not item.get("url"):
                raise ValueError("其它資料庫自訂策略須提供 url")
            plan.append(item)
        return plan
    names = field_terms(cfg["instrument_terms"])
    oral = '("Oral Health"[Mesh] OR ' + field_terms(cfg["oral_terms"]) + ')'
    efficacy = '("Self Efficacy"[Mesh] OR ' + field_terms(cfg["efficacy_terms"]) + ')'
    method = '("Psychometrics"[Mesh] OR "Surveys and Questionnaires"[Mesh] OR ' + field_terms(cfg["method_terms"]) + ')'
    pop = '("Aged"[Mesh] OR "Aged, 80 and over"[Mesh] OR ' + field_terms(cfg["population_terms"]) + ')'
    translation = field_terms(cfg["translation_terms"])
    context = field_terms(cfg["context_terms"])
    locale = field_terms(cfg["locale_terms"])
    concept = f"({oral} AND {efficacy})"
    pm = [
        ("names", "GSEOH 名稱追蹤（不限制翻譯、年齡或社區）", names),
        ("methods_broad", "口腔自我效能量表與測量研究（敏感性補查）", f"{concept} AND {method}"),
        ("methods_population", "老人相關量表與測量研究", f"{concept} AND {method} AND {pop}"),
        ("translations", "口腔自我效能翻譯與文化調適（不限年齡）", f"{concept} AND {translation}"),
        ("community", "社區老人聚焦（不得替代廣搜）", f"{concept} AND {method} AND {pop} AND {context}"),
        ("locale", "臺灣與華語版本補查（語言版本非出版語言）", f"({names} OR ({concept} AND {method})) AND {locale}"),
    ]
    result = []
    for key, label, term in pm:
        result.append(dict(database="PubMed / NCBI E-utilities", id=key, label=label, query=term,
                           url="https://pubmed.ncbi.nlm.nih.gov/?" + urllib.parse.urlencode({"term": term}), status="尚未執行"))
    lines = [
        '#1 (GSEOH OR "Geriatric Self-Efficacy Scale for Oral Health"):ti,ab,kw',
        '#2 (oral OR dental):ti,ab,kw',
        '#3 ("self efficacy" OR "self-efficacy"):ti,ab,kw',
        '#4 #2 AND #3',
        '#5 (scale* OR questionnaire* OR instrument* OR psychometr* OR valid* OR reliab* OR translat* OR "cross-cultural" OR "cultural adaptation"):ti,ab,kw',
        '#6 (elderly OR geriatric* OR aged OR "older adult*" OR "older people"):ti,ab,kw',
        '#7 #1 OR (#4 AND #5)',
        '#8 #4 AND #6',
    ]
    result.append(dict(database="Cochrane Library", id="cochrane", label="逐行貼入 Search Manager；CDSR 與 CENTRAL 分別紀錄", query="\n".join(lines),
                       url="https://www.cochranelibrary.com/advanced-search/search-manager", status="待網頁搜尋／RIS 匯入"))
    scholar = [
        ('s1', '原量表全文與被引用追蹤', '"Development of an oral health-related self-efficacy scale for use with older adults"'),
        ('s2', '縮寫廣搜', 'GSEOH'),
        ('s3', '翻譯', 'GSEOH translation'),
        ('s4', '文化調適', 'GSEOH "cross-cultural"'),
        ('s5', '華語版補查', 'GSEOH Chinese'),
        ('s6', '臺灣版補查', 'GSEOH Taiwan'),
        ('s7', '中文題名補查', '"老年人口腔健康" "自我效能" "汉化"'),
        ('s8', '中文題名另一寫法', '"口腔健康相关自我效能" "信效度"'),
        ('s9', '同構念量表', '"oral health" "self-efficacy" scale older'),
        ('s10', '日文來源補查', '"口腔" "自己効力感" "高齢者"'),
    ]
    for key, label, term in scholar:
        result.append(dict(database="Google Scholar", id=key, label=label, query=term,
                           url="https://scholar.google.com/scholar?" + urllib.parse.urlencode({"q": term}), status="待網頁搜尋／RIS 匯入"))
    return result


def write_plan(out, cfg, plan):
    save_json(out / "search_plan.json", dict(created_at=stamp(), config=cfg, searches=plan))
    cards = []
    for p in plan:
        cards.append(f'<article><small>{html.escape(p["database"])}</small><h2>{html.escape(p["label"])}</h2>'
                     f'<pre>{html.escape(p["query"])}</pre><button onclick="copy(this)">複製搜尋式</button> '
                     f'<a target="_blank" rel="noopener" href="{html.escape(p["url"], quote=True)}">開啟資料庫</a></article>')
    page = '''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>文獻搜尋式</title><style>body{font:16px/1.7 system-ui;background:#f5f4ef;color:#203238;max-width:1000px;margin:40px auto;padding:0 24px}h1{font-size:30px}h2{font-size:19px}article{background:white;border:1px solid #d7dedb;border-radius:8px;padding:24px;margin:20px 0}small{color:#45756f}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 monospace;background:#f1f5f4;padding:18px}a,button{color:#18534b}button{padding:7px 12px;cursor:pointer}details{margin:20px 0}footer{padding:24px 0;color:#52635f}</style>
<h1>文獻搜尋式</h1><p>依本次研究問題建立主要與補充搜尋，保留各平台語法及執行狀態。</p>
<p>PubMed 可由工具自動搜尋。Cochrane 與 Scholar 在網頁執行後匯出 RIS；此頁的連結與搜尋式不表示已完成搜尋。</p>'''
    page = page.replace('<title>文獻搜尋式</title>', '<title>' + html.escape(cfg["topic"]) + '</title>').replace('<h1>文獻搜尋式</h1>', '<h1>' + html.escape(cfg["topic"]) + '</h1>')
    context_data = {"PICO": cfg["pico"]}
    if cfg.get("author_correspondence"):
        context_data["作者條件"] = cfg["author_correspondence"]
    page += "<details><summary>本次研究條件</summary><pre>" + html.escape(json.dumps(context_data, ensure_ascii=False, indent=2)) + "</pre></details>"
    page += "\n".join(cards)
    page += '''<footer>候選文獻仍須依本次納入條件篩選。Google Scholar 可從重要原文的「被引用」及「所有版本」補查。搜尋語法來源：NLM PubMed User Guide、Cochrane Handbook、Google Scholar Search Help。</footer>
<script>async function copy(b){const t=b.parentElement.querySelector('pre').textContent;try{await navigator.clipboard.writeText(t);b.textContent='已複製'}catch{const x=document.createElement('textarea');x.value=t;document.body.appendChild(x);x.select();document.execCommand('copy');x.remove();b.textContent='已複製'}}</script></html>'''
    (out / "搜尋式.html").write_text(page, encoding="utf-8")


class Client:
    def __init__(self, cache):
        self.cache = cache
        self.last = 0.0

    def get(self, url, params=None):
        if params:
            url += "?" + urllib.parse.urlencode(params)
        error = None
        for attempt in range(3):
            time.sleep(max(0, .42 - (time.monotonic() - self.last)))
            self.last = time.monotonic()
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "GSEOH-Literature-Tool/1.0", "Accept": "*/*"})
                request_started = time.monotonic()
                with urllib.request.urlopen(request, timeout=15) as response:
                    chunks, size = [], 0
                    while True:
                        chunk = response.read1(65536)
                        if not chunk:
                            break
                        chunks.append(chunk)
                        size += len(chunk)
                        if size > 60 * 1024 * 1024:
                            raise ValueError("回應大於 60 MB，停止下載")
                        if time.monotonic() - request_started > 45:
                            raise TimeoutError("回應下載超過 45 秒")
                    data = b"".join(chunks)
                if len(data) > 60 * 1024 * 1024:
                    raise ValueError("回應大於 60 MB，停止下載")
                return data
            except urllib.error.HTTPError as exc:
                error = exc
                if exc.code not in (429, 500, 502, 503, 504):
                    raise
                time.sleep(min(30, int(exc.headers.get("Retry-After", "0"))) if exc.headers.get("Retry-After", "0").isdigit() else 2 ** attempt)
            except (urllib.error.URLError, TimeoutError) as exc:
                error = exc
                time.sleep(2 ** attempt)
        raise RuntimeError(f"網路請求失敗：{type(error).__name__}") from error

    def eutils(self, service, **params):
        params.update(tool="GSEOH_Literature_Tool")
        if os.environ.get("NCBI_EMAIL"):
            params["email"] = os.environ["NCBI_EMAIL"]
        if os.environ.get("NCBI_API_KEY"):
            params["api_key"] = os.environ["NCBI_API_KEY"]
        data = self.get(EUTILS + service + ".fcgi", params)
        safe_params = {k: v for k, v in params.items() if k not in ("api_key", "email")}
        key = hashlib.sha256(json.dumps(safe_params, sort_keys=True).encode()).hexdigest()[:16]
        (self.cache / f"{service}_{key}.raw").write_bytes(data)
        return data


def node_text(node):
    return " ".join("".join(node.itertext()).split()) if node is not None else ""


def parse_pubmed(data):
    root = ET.fromstring(data)
    error = root.find(".//ERROR")
    if error is not None:
        raise ValueError(node_text(error))
    records = []
    for entry in root.findall("PubmedArticle"):
        article = entry.find("MedlineCitation/Article")
        ids = {x.get("IdType"): node_text(x) for x in entry.findall("PubmedData/ArticleIdList/ArticleId")}
        pubdate = article.find("Journal/JournalIssue/PubDate")
        date_text = node_text(pubdate.find("Year")) or node_text(pubdate.find("MedlineDate")) if pubdate is not None else ""
        year_match = re.search(r"\b(?:19|20)\d{2}\b", date_text)
        authors = []
        for a in article.findall("AuthorList/Author"):
            authors.append(node_text(a.find("CollectiveName")) or " ".join(filter(None, [node_text(a.find("LastName")), node_text(a.find("Initials"))])))
        abstract = "\n".join((x.get("Label", "") + ": " if x.get("Label") else "") + node_text(x) for x in article.findall("Abstract/AbstractText"))
        pmid = node_text(entry.find("MedlineCitation/PMID"))
        records.append(dict(title=node_text(article.find("ArticleTitle")), year=int(year_match.group()) if year_match else None,
                            authors="; ".join(authors), journal=node_text(article.find("Journal/Title")),
                            pmid=pmid, doi=ids.get("doi", ""), pmcid=ids.get("pmc", ""), abstract=abstract,
                            url="https://pubmed.ncbi.nlm.nih.gov/" + pmid + "/", source="PubMed / E-utilities",
                            query_ids=[], publication_types=[node_text(x) for x in article.findall("PublicationTypeList/PublicationType")]))
    for entry in root.findall("PubmedBookArticle"):
        doc = entry.find("BookDocument")
        pmid = node_text(doc.find("PMID"))
        records.append(dict(title=node_text(doc.find("ArticleTitle")), pmid=pmid, source="PubMed / E-utilities", query_ids=[],
                            url="https://pubmed.ncbi.nlm.nih.gov/" + pmid + "/", kind="書籍章節（需篩選）"))
    return records


def norm_doi(value):
    return re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", value.strip(), flags=re.I).lower()


def same_record(a, b):
    if a.get("pmid") and b.get("pmid"):
        return str(a["pmid"]) == str(b["pmid"])
    if a.get("doi") and b.get("doi"):
        return norm_doi(a["doi"]) == norm_doi(b["doi"])
    normalize = lambda s: re.sub(r"[\W_]", "", s.casefold())
    return bool(a.get("title") and b.get("title") and a.get("year") and a.get("year") == b.get("year")
                and normalize(a["title"]) == normalize(b["title"]))


def merge_records(records):
    result = []
    for record in records:
        if record.get("doi"):
            record["doi"] = norm_doi(record["doi"])
        existing = next((r for r in result if same_record(r, record)), None)
        if existing is None:
            result.append(dict(record))
        else:
            for key, value in record.items():
                if key in ("query_ids", "publication_types"):
                    existing[key] = list(dict.fromkeys(existing.get(key, []) + value))
                elif key == "source":
                    existing[key] = "; ".join(dict.fromkeys(existing.get(key, "").split("; ") + value.split("; ")))
                elif not existing.get(key) and value:
                    existing[key] = value
    return result


def parse_ris(path, source, query_id):
    records, current, last = [], {}, ""
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        match = re.match(r"^([A-Z0-9]{2})\s{2}-\s?(.*)$", line)
        if match:
            key, value = match.groups()
            if key == "TY":
                current = {}
            if key == "ER":
                title = (current.get("TI") or current.get("T1") or [""])[0]
                if title:
                    date = (current.get("PY") or current.get("Y1") or [""])[0]
                    year = re.search(r"\b(?:19|20)\d{2}\b", date)
                    records.append(dict(title=title, year=int(year.group()) if year else None,
                                        authors="; ".join(current.get("AU", current.get("A1", []))),
                                        journal=(current.get("JO") or current.get("JF") or current.get("T2") or [""])[0],
                                        doi=(current.get("DO") or [""])[0], url=(current.get("UR") or [""])[0],
                                        abstract="\n".join(current.get("AB", [])), source=source, query_ids=[query_id]))
                current = {}
            else:
                current.setdefault(key, []).append(value)
            last = key
        elif line.strip() and last in current:
            current[last][-1] += " " + line.strip()
    if current:
        raise ValueError("RIS 檔最後一筆缺少 ER 結尾；請重新匯出")
    return records


def classify(r):
    if r.get("kind"):
        return r["kind"]
    text = (r.get("title", "") + " " + r.get("abstract", "")).lower()
    if "gseoh" in text or "27531046" == r.get("pmid"):
        return "GSEOH 名稱或原文命中（仍需閱讀）"
    return "相關量表／方法候選（需人工篩選）"


def s3_https(url):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "s3" and parsed.netloc == "pmc-oa-opendata":
        return CLOUD + parsed.path.lstrip("/") + ("?" + parsed.query if parsed.query else "")
    if parsed.scheme == "https" and parsed.netloc == "pmc-oa-opendata.s3.amazonaws.com":
        return url
    raise ValueError("PMC metadata PDF URL 不屬於官方資料 bucket")


def pdf_bytes_ok(data):
    return len(data) > 100 and data[:1024].lstrip().startswith(b"%PDF-")


def existing_pdf_path(record, out=None):
    """Return a readable PDF, resolving manifest paths beside the manifest."""
    candidates = []
    if record.get("local_pdf"):
        candidates.append((record["local_pdf"], ROOT))
    if record.get("pdf_path"):
        candidates.append((record["pdf_path"], Path(out) if out is not None else ROOT))
    for value, base in candidates:
        try:
            path = Path(value)
            path = path if path.is_absolute() else base / path
            if path.is_file():
                with path.open("rb") as file:
                    if pdf_bytes_ok(file.read(1024)):
                        return path.resolve()
        except (OSError, TypeError, ValueError):
            continue
    return None


_MISSING_PDF_STATUS = "本地 PDF 不存在、不可讀或檔頭無效，待重新取得"


def refresh_pdf_paths(records, out):
    """Keep usable paths and retain stale locations without marking them ready."""
    for record in records:
        existing = existing_pdf_path(record, out)
        if existing is not None:
            record["pdf_path"] = str(existing)
            if record.get("pdf_status") == _MISSING_PDF_STATUS:
                record["pdf_status"] = "已有本地 PDF（已恢復）"
        elif record.get("pdf_path"):
            record["pdf_missing_path"] = record.pop("pdf_path")
            record["pdf_status"] = _MISSING_PDF_STATUS


def download_pmc(r, client, out):
    pmcid = r["pmcid"]
    if not re.fullmatch(r"PMC\d+", pmcid):
        raise ValueError("PMCID 格式錯誤")
    listing = ET.fromstring(client.get(CLOUD, {"list-type": "2", "prefix": pmcid + ".", "delimiter": "/"}))
    prefixes = [node_text(p) for p in listing.findall("{*}CommonPrefixes/{*}Prefix")]
    candidates = []
    for prefix in prefixes:
        if not re.fullmatch(re.escape(pmcid) + r"\.\d+/", prefix):
            continue
        key = prefix.rstrip("/")
        meta_url = CLOUD + prefix + key + ".json"
        meta = json.loads(client.get(meta_url))
        save_json(out / "raw" / (key + ".json"), meta)
        if meta.get("pdf_url"):
            candidates.append((meta, meta_url))
    if not candidates:
        r["pdf_status"] = "PMC 雲端無可下載 PDF（不代表網頁無全文）"
        return
    # Prefer a published version. Higher deposited version is only a tie-breaker.
    meta, meta_url = sorted(candidates, key=lambda m: (str(m[0].get("is_manuscript", "")).lower() in ("yes", "true"), -int(m[0].get("version", 1))))[0]
    url = s3_https(meta["pdf_url"])
    data = client.get(url)
    if not pdf_bytes_ok(data):
        raise ValueError("回應不是有效 PDF 檔頭")
    digest = urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get("md5", [""])[0]
    if digest and hashlib.md5(data).hexdigest().lower() != digest.lower():
        raise ValueError("PMC PDF MD5 校驗不符")
    title = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", r["title"]).strip(" .")[:75]
    file = out / "pdfs" / f'{r.get("year") or "undated"}_{pmcid}_{title}.pdf'
    file.write_bytes(data)
    r.update(pdf_status="已下載（PMC 官方雲端）", pdf_path=str(file.resolve()), pdf_url=url,
             license=str(meta.get("license_code", "")), pmc_version=str(meta.get("version", "")),
             pdf_sha256=hashlib.sha256(data).hexdigest(), metadata_url=meta_url)


def prepare_pdfs(records, client, out, limit):
    attempts = 0
    refresh_pdf_paths(records, out)
    for r in records:
        r.setdefault("fulltext_url", "https://pmc.ncbi.nlm.nih.gov/articles/" + r["pmcid"] + "/" if r.get("pmcid") else ("https://doi.org/" + r["doi"] if r.get("doi") else r.get("url", "")))
        existing = existing_pdf_path(r, out)
        if existing is not None:
            if r.get("local_pdf"):
                r.update(pdf_status="已有本地 PDF（先前取得）", pdf_path=str(existing), pdf_sha256=hashlib.sha256(existing.read_bytes()).hexdigest())
            continue
        elif r.get("pmcid") and attempts < limit:
            attempts += 1
            print(f'PDF {attempts}/{limit}: {r["pmcid"]}', flush=True)
            try:
                download_pmc(r, client, out)
            except Exception as exc:
                r["pdf_status"] = "下載失敗／需重試"
                r["pdf_error"] = str(exc)
        elif r.get("pmcid"):
            r["pdf_status"] = "尚未下載（本次下載上限）"
        else:
            r["pdf_status"] = "未自動取得：至期刊／機構或作者查詢"


HEADERS = ["序號", "文章標題", "年份", "作者", "期刊", "文獻類別", "PMID", "DOI", "PMCID", "來源", "命中搜尋式", "書目或官方網址", "全文入口", "PDF 狀態", "PDF 本地路徑", "PDF 來源網址", "PDF 授權", "摘要節錄", "篩選決定", "排除理由", "閱讀筆記"]


def record_row(i, r):
    return [i, r.get("title", ""), r.get("year"), r.get("authors", ""), r.get("journal", ""), classify(r),
            r.get("pmid", ""), r.get("doi", ""), r.get("pmcid", ""), r.get("source", ""), "; ".join(r.get("query_ids", [])),
            r.get("url", ""), r.get("fulltext_url", ""), r.get("pdf_status", ""), r.get("pdf_path", ""), r.get("pdf_url", ""),
            r.get("license", ""), re.sub(r"\s+", " ", r.get("abstract", ""))[:240], r.get("screening_decision", "未篩選"), r.get("exclusion_reason", ""), r.get("reading_notes", "")]


def write_reading_start(out, cfg, records, logs, plan, excel_ready=False):
    """Provide a researcher-facing entry point using only available artifacts."""
    out = Path(out).resolve()

    def text(value):
        return html.escape(str(value) if value is not None else "")

    def link(label, url):
        return f'<a href="{html.escape(url, quote=True)}">{text(label)}</a>'

    cards = []
    if excel_ready and (out / "文獻清單.xlsx").is_file():
        cards.append(link("開啟 Excel 閱讀清單", urllib.parse.quote("文獻清單.xlsx")))
    if (out / "文獻清單.csv").is_file():
        cards.append(link("開啟文獻清單（CSV）", urllib.parse.quote("文獻清單.csv")))
    if (out / "搜尋式.html").is_file():
        cards.append(link("查看搜尋策略", urllib.parse.quote("搜尋式.html")))
    available = 0
    rows = []
    for record in records:
        pdf = existing_pdf_path(record, out)
        if pdf is not None:
            available += 1
            try:
                url = urllib.parse.quote(pdf.relative_to(out).as_posix())
            except ValueError:
                url = pdf.as_uri()
            action = link("閱讀 PDF", url)
            state = "全文檔案可用（仍需核對內容與版本）"
        else:
            url = record.get("fulltext_url") or record.get("url") or ""
            if not isinstance(url, str):
                url = ""
            try:
                parsed = urllib.parse.urlsplit(url)
            except ValueError:
                parsed = urllib.parse.urlsplit("")
            action = (link("查看文獻入口", url)
                      if parsed.scheme in {"https", "http"} and parsed.netloc
                      else "待查可取得來源")
            state = "待取得全文"
        note = record.get("reading_notes")
        notes = f'<details><summary>閱讀筆記</summary><p>{text(note)}</p></details>' if note else ""
        rows.append(f'<tr><td>{text(record.get("title"))}{notes}</td>'
                    f'<td>{text(record.get("year"))}</td><td>{state}</td>'
                    f'<td>{text(record.get("screening_decision") or "未篩選")}</td>'
                    f'<td>{action}</td></tr>')
    executed = {log.get("id") for log in logs}
    pending = [item for item in plan if item.get("id") not in executed]
    failed = sum("失敗" in log.get("status", "") for log in logs)
    truncated = sum(bool(log.get("truncated")) for log in logs)
    notices = []
    if failed:
        notices.append(f"{failed} 輪搜尋或取回未成功，候選清單可能不完整。")
    if pending:
        notices.append(f"{len(pending)} 輪搜尋尚未執行，尚不能視為完整回顧。")
    if truncated:
        notices.append(f"{truncated} 輪搜尋只取回部分結果。")
    if any(log.get("warnings") for log in logs):
        notices.append("部分搜尋有警告，需核對其對涵蓋範圍的影響。")
    if not excel_ready:
        notices.append("本次 Excel 匯出尚未完成；可先使用已有清單與全文。")
    if not records:
        notices.append("目前沒有可閱讀的候選文獻；不能由此推論沒有相關研究。")
    checks = []
    for item in logs + pending:
        checks.append('<li>' + text(item.get("database") or "來源待確認") + '：'
                      + text(item.get("label") or item.get("id") or "搜尋紀錄") + ' — '
                      + text(item.get("status") if item in logs else "尚未執行") + '</li>')
    body = f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>開始閱讀 — {text(cfg.get("topic"))}</title>
<style>body{{font-family:system-ui,sans-serif;background:#f5f7f6;color:#243b3b;margin:0}}
main{{max-width:1100px;margin:auto;padding:32px 20px}}h1{{line-height:1.4}}
nav{{display:flex;flex-wrap:wrap;gap:12px;margin:24px 0}}a{{color:#17696b}}
nav a{{background:white;padding:14px 18px;border:1px solid #ccdada;border-radius:8px}}
.notice{{background:#fff4d8;padding:16px;border-radius:8px}}table{{border-collapse:collapse;width:100%;background:white}}
th,td{{text-align:left;padding:14px;border-bottom:1px solid #dce5e1;vertical-align:top}}th{{background:#e7efec}}
.table{{overflow-x:auto}}details{{margin:20px 0}}td details{{margin:10px 0;font-size:.9em}}td p{{white-space:pre-wrap}}
</style><main><h1>開始閱讀</h1><p>{text(cfg.get("topic"))}</p>
<p>已整理 {len(records)} 筆候選文獻；{available} 筆全文檔案可用，{len(records) - available} 筆待取得全文。</p>
<p>先打開閱讀清單或下方全文，記下與研究問題的關係。這是候選清單，尚未代表完成納入、全文評讀或品質評級。</p>
<nav>{' '.join(cards)}</nav>
{('<div class="notice">' + '<br>'.join(notices) + '</div>') if notices else ''}
<h2>候選閱讀清單</h2><div class="table"><table><thead><tr><th>文章</th><th>年份</th><th>全文</th><th>篩選決定</th><th>開啟</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<details><summary>查看本次搜尋範圍與進度</summary><ul>{''.join(checks)}</ul></details>
</main></html>'''
    (out / "開始閱讀.html").write_text(body, encoding="utf-8")


def export(out, cfg, records, logs, plan):
    if "GSEOH" not in cfg.get("instrument_terms", []):
        for record in records:
            record.setdefault("kind", "候選文獻（需人工篩選）")
    records.sort(key=lambda r: (0 if r.get("pmid") == cfg.get("seed_pmid") else 1 if "GSEOH" in classify(r) else 2, -(r.get("year") or 0), r.get("title", "")))
    payload = dict(created_at=stamp(), topic=cfg["topic"], pico=cfg["pico"], records=records, searches=logs,
                   strategy=plan, source_attribution="Bibliographic data: NLM PubMed. PDFs: NIH NLM NCBI PMC Article Datasets on AWS, accessed " + stamp())
    if cfg.get("missing_pdf_priority"):
        payload["missing_pdf_priority"] = cfg["missing_pdf_priority"]
        payload["ranking_note"] = cfg.get("ranking_note", "")
    save_json(out / "manifest.json", payload)
    with (out / "文獻清單.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS)
        writer.writerows([("'" + v if isinstance(v, str) and v.startswith(("=", "+", "-", "@")) else v) for v in record_row(i, r)] for i, r in enumerate(records, 1))
    write_reading_start(out, cfg, records, logs, plan)
    node = os.environ.get("CODEX_NODE") or shutil.which("node")
    bundled = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe"
    if not node and bundled.exists():
        node = str(bundled)
    if not node:
        print("Excel 匯出未完成：找不到 Node。閱讀入口、JSON 與 CSV 已保存。", flush=True)
        return False
    try:
        completed = subprocess.run([node, str(HERE / "export_excel.mjs"), str(out / "manifest.json"), str(out / "文獻清單.xlsx")], cwd=HERE)
    except OSError:
        print("Excel 匯出未完成：無法啟動 Node。閱讀入口、JSON 與 CSV 已保存。", flush=True)
        return False
    write_reading_start(out, cfg, records, logs, plan, excel_ready=completed.returncode == 0)
    return completed.returncode == 0


def enrich_citations(records, client, out):
    """Match exact DOIs only; missing coverage and request failures are not zero."""
    targets = [r for r in records if existing_pdf_path(r, out) is None and r.get("doi")]
    by_doi = {norm_doi(r["doi"]): r for r in targets}
    dois = list(by_doi)
    for offset in range(0, len(dois), 50):
        batch = dois[offset:offset + 50]
        params = {"filter": "doi:" + "|".join(batch), "per_page": 100,
                  "select": "id,doi,display_name,cited_by_count,publication_year"}
        if os.environ.get("OPENALEX_API_KEY"):
            params["api_key"] = os.environ["OPENALEX_API_KEY"]
        checked = stamp()
        try:
            response = json.loads(client.get("https://api.openalex.org/works", params))
            save_json(out / "raw" / f"openalex_citations_{offset}.json", response)
            for work in response.get("results", []):
                record = by_doi.get(norm_doi(work.get("doi") or ""))
                if record is not None:
                    record.update(citation_count=work.get("cited_by_count"), citation_source="OpenAlex", citation_checked_at=checked,
                                  citation_url=work["id"], citation_status="DOI 精確匹配")
            for doi in batch:
                by_doi[doi].setdefault("citation_status", "OpenAlex 本次未匹配（非零引用）")
                by_doi[doi].setdefault("citation_checked_at", checked)
        except Exception as exc:
            for doi in batch:
                by_doi[doi].update(citation_status="引用數取得失敗（非零引用）", citation_error=str(exc), citation_checked_at=checked)
        print(f"引用數查核：{min(offset + 50, len(dois))}/{len(dois)} DOI", flush=True)
    for r in records:
        if existing_pdf_path(r, out) is None:
            r.setdefault("citation_status", "缺 DOI，待人工核對標題與年份")
            r.setdefault("quality_status", "尚未取得全文，COSMIN 待評；不以引用數推定品質")


def missing_pdf_priority(records, limit=20, quality_property="", core_terms=None, out=None):
    core_terms = [t.casefold() for t in (core_terms if core_terms is not None else ["GSEOH"]) if t.strip()]
    def core_record(record):
        text = " ".join(str(record.get(k, "")) for k in ("title", "abstract", "kind")).casefold()
        return any(term in text for term in core_terms)
    order = {"very good": 0, "adequate": 1, "doubtful": 2, "inadequate": 3}
    def comparable_assessment(record):
        return next((a for a in record.get("quality_assessments", []) if a.get("property", "").casefold() == quality_property.casefold()
                     and a.get("rating", "").casefold() in order and all(a.get(k) for k in ("source", "location", "tool"))), None) if quality_property else None
    def quality_key(record):
        a = comparable_assessment(record)
        return (0, order[a["rating"].casefold()]) if a else (1, 0)
    unavailable = [r for r in records if existing_pdf_path(r, out) is None]
    unavailable.sort(key=lambda r: (0 if core_record(r) else 1, *quality_key(r), r.get("citation_count") is None,
                                     -(r.get("citation_count") or 0), -(r.get("year") or 0)))
    rows = []
    for i, r in enumerate(unavailable[:limit], 1):
        assessment = comparable_assessment(r)
        rows.append(dict(priority=i, title=r["title"], year=r.get("year"), doi=r.get("doi", ""), pmid=r.get("pmid", ""),
                         scope="指定量表／核心名稱命中，優先取得" if core_record(r) else "研究問題候選（需篩選）",
                         citation_count=r.get("citation_count"), citation_source=r.get("citation_source", ""),
                         citation_checked_at=r.get("citation_checked_at", ""), citation_url=r.get("citation_url", ""),
                         citation_status=r.get("citation_status", "尚未查核"),
                         quality_status="已評指定特性（見證據欄）" if assessment else r.get("quality_status", "COSMIN 待評"),
                         quality_property=quality_property, quality_rating=assessment["rating"] if assessment else "待評／尚無可比較評級",
                         quality_assessments=r.get("quality_assessments", []),
                         fulltext_url=r.get("fulltext_url", r.get("url", "")), pdf_status=r.get("pdf_status", "未取得")))
    return rows


def run_search(args, cfg, out, plan):
    client = Client(out / "raw")
    records, logs, query_members = [], [], {}
    selected = [p for p in plan if p["database"].startswith("PubMed") and p["id"] in args.queries]
    if len(selected) != len(set(args.queries)):
        raise ValueError("--queries 含未定義的搜尋式")
    for p in selected:
        print(f'搜尋 {p["id"]}: {p["label"]}', flush=True)
        log = dict(p, searched_at=stamp())
        try:
            result = json.loads(client.eutils("esearch", db="pubmed", term=p["query"], retmode="json", retmax=args.max_per_query, sort="relevance"))["esearchresult"]
            errors = result.get("errorlist", {})
            if errors.get("fieldsnotfound") or errors.get("phrasesnotfound"):
                log["warnings"] = errors
            if result.get("ERROR"):
                raise ValueError(result["ERROR"])
            ids = result.get("idlist", [])
            count = int(result["count"])
            log.update(total_hits=count, retrieved=len(ids), truncated=count > len(ids), translated_query=result.get("querytranslation", ""), warnings=result.get("warninglist", errors), status="已執行")
            query_members[p["id"]] = ids
            for offset in range(0, len(ids), 150):
                batch = parse_pubmed(client.eutils("efetch", db="pubmed", id=",".join(ids[offset:offset + 150]), retmode="xml"))
                for record in batch:
                    record["query_ids"] = [p["id"]]
                records.extend(batch)
            if len(ids) != sum(1 for r in records if p["id"] in r.get("query_ids", [])):
                raise ValueError("書目取回筆數與 PMID 清單不一致，請檢查 raw API 回應")
            print(f'命中 {count}；取回 {len(ids)}' + ('（有截取，非完整檢索）' if count > len(ids) else ''), flush=True)
        except Exception as exc:
            log.update(status="搜尋或取回失敗（非零命中）", error=str(exc))
            print(f'失敗：{exc}', flush=True)
        logs.append(log)
    seed = cfg.get("seed_pmid")
    covered = [key for key, ids in query_members.items() if seed in ids]
    if seed and not any(r.get("pmid") == seed for r in records):
        try:
            seed_records = parse_pubmed(client.eutils("efetch", db="pubmed", id=seed, retmode="xml"))
            for r in seed_records:
                r["query_ids"] = ["seed_manual"]
            records.extend(seed_records)
            logs.append(dict(database="PubMed / NCBI E-utilities", id="seed_manual", label="原始種子直接補取（不是搜尋命中）", query=f"{seed}[uid]", status="直接補取", retrieved=len(seed_records), searched_at=stamp()))
        except Exception as exc:
            logs.append(dict(id="seed_manual", status="失敗", error=str(exc)))
    records += cfg.get("supplemental_records", [])
    records = merge_records(records)
    prepare_pdfs(records, client, out, args.pdf_limit)
    success = export(out, cfg, records, logs, plan)
    summary = dict(records=len(records), new_pdf_count=sum(r.get("pdf_status") == "已下載（PMC 官方雲端）" for r in records),
                   existing_pdf_count=sum(r.get("pdf_status") == "已有本地 PDF（先前取得）" for r in records),
                   failed_queries=[l.get("id") for l in logs if "失敗" in l.get("status", "")],
                   truncated_queries=[l.get("id") for l in logs if l.get("truncated")], seed_coverage=covered, excel_exported=success,
                   note="候選清單未經納入篩選；人工補入項目依各筆實際來源標示。尚未執行 Cochrane／Scholar 網頁檢索。")
    save_json(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if success and not summary["failed_queries"] else 2


def main():
    parser = argparse.ArgumentParser(description="PICO 文獻搜尋與閱讀清單工具")
    parser.add_argument("action", choices=["plan", "run", "import", "rank"])
    default_config = HERE / "config_gseoh.json" if (HERE / "config_gseoh.json").exists() else HERE.parent / "assets" / "gseoh_profile.json"
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--queries", nargs="+", default=["names", "methods_population", "translations"])
    parser.add_argument("--max-per-query", type=int, default=500)
    parser.add_argument("--pdf-limit", type=int, default=20)
    parser.add_argument("--ris", type=Path)
    parser.add_argument("--source", default="手動匯入（來源未指定）")
    parser.add_argument("--query-id", default="manual_import")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--quality-property", default="")
    args = parser.parse_args()
    if not 1 <= args.max_per_query <= 9999 or args.pdf_limit < 0:
        parser.error("max-per-query 應為 1–9999；pdf-limit 應大於等於 0")
    cfg = json.loads(args.config.read_text(encoding="utf-8-sig"))
    default_name = "literature_" + datetime.now(TAIPEI).strftime("%Y%m%d_%H%M%S")
    out = (args.out or ROOT / "outputs" / default_name).resolve()
    if args.action == "run" and (out / "manifest.json").exists():
        parser.error("輸出目錄已有搜尋成果，請另指定新目錄，避免覆蓋閱讀筆記")
    out.mkdir(parents=True, exist_ok=True)
    for folder in ["raw", "pdfs"]:
        (out / folder).mkdir(exist_ok=True)
    plan = make_plan(cfg)
    if args.action == "plan":
        write_plan(out, cfg, plan)
        print(out / "搜尋式.html")
        return 0
    if args.action == "rank":
        if not (out / "manifest.json").is_file():
            parser.error("rank 須以 --out 指定已有 manifest 的搜尋目錄")
        payload = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        records = payload["records"]
        refresh_pdf_paths(records, out)
        enrich_citations(records, Client(out / "raw"), out)
        priority = missing_pdf_priority(records, args.top, args.quality_property, cfg.get("instrument_terms", []), out=out)
        save_json(out / "待取得全文優先清單.json", priority)
        cfg["missing_pdf_priority"] = priority
        cfg["ranking_note"] = "指定核心文獻保留；同一指定特性的已評組按品質排序，其餘依同來源引用次數規劃取得。無可靠全文評估者 COSMIN 待評，不宣稱品質最高。"
        success = export(out, cfg, records, payload["searches"], payload["strategy"])
        current = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        current["missing_pdf_priority"] = priority
        current["ranking_note"] = cfg["ranking_note"]
        save_json(out / "manifest.json", current)
        return 0 if success else 2
    if args.action == "import":
        if not args.ris or not (out / "manifest.json").is_file():
            parser.error("匯入須指定 --ris 與含 manifest.json 的既有 --out")
        payload = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        imported = parse_ris(args.ris, args.source, args.query_id)
        records = merge_records(payload["records"] + imported)
        logs = payload["searches"] + [dict(database=args.source, id=args.query_id, query="匯入，原搜尋式須另保留", searched_at=stamp(), retrieved=len(imported), status="RIS 手動匯入")]
        prepare_pdfs(records, Client(out / "raw"), out, 0)
        return 0 if export(out, cfg, records, logs, payload["strategy"]) else 2
    write_plan(out, cfg, plan)
    return run_search(args, cfg, out, plan)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None
    sys.exit(main())
