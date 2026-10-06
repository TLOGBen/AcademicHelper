# PICO literature search in Codex Cloud

The skill lives in `.agents/skills/pico-literature-search/`. Repository skills are available in Codex Cloud; personal computer skills are not automatically synced. See [OpenAI Cloud environments](https://learn.chatgpt.com/docs/environments/cloud-environments) and [Build skills](https://learn.chatgpt.com/docs/build-skills).

## Create the environment

1. Choose **Work in > Cloud > Select environment > Create environment**, or open **Settings > Codex Cloud > Environments**.
2. Select `TLOGBen/AcademicHelper`. For the proposed version in PR #1, ask setup to load `codex/pico-literature-cloud`; after review and merge, the default branch can be used instead. Do not assume an unmerged PR is already in `main`.
3. Ask setup to prepare Python 3.11+, Node, and the available spreadsheet runtime. The new Python helper uses only the standard library; the existing AcademicHelper MCP server has its own dependencies in `pyproject.toml`.
4. Allow the API hosts `eutils.ncbi.nlm.nih.gov`, `pmc-oa-opendata.s3.amazonaws.com`, and `api.openalex.org`. Add official publication hosts as needed for source research. Package-manager-only access is insufficient for literature API requests.
5. Run the checks below, review the setup report, and publish only after its required outputs are verified. Publishing captures the prepared environment for new tasks.

Suggested setup prompt:

> Prepare this repository for the pico-literature-search skill. Find the available Python, Node, and spreadsheet runtime, and configure their actual paths. Verify NCBI E-utilities, the official PMC open-data bucket, and OpenAlex access. Run the helper tests and offline plan. Then perform a bounded GSEOH name search with at most 20 records and one PDF attempt. Check the original PMID 27531046, any downloaded PDF, and the exported XLSX. If the Excel runtime is unavailable, keep JSON/CSV, report the unfinished XLSX requirement, and use the available spreadsheet capability to finish it. Record unexecuted Cochrane and Scholar searches. Do not publish a setup with unmet requirements.

## Local or cloud checks

Run from the repository root:

```text
python -B -m unittest discover -s .agents/skills/pico-literature-search/scripts -p test_literature_tool.py -v
python .agents/skills/pico-literature-search/scripts/literature_tool.py plan --out outputs/plan_check
python .agents/skills/pico-literature-search/scripts/literature_tool.py run --out outputs/cloud_smoke --queries names --max-per-query 20 --pdf-limit 1
```

The first two commands are offline. The last uses NCBI and PMC and should run once the environment's network access is configured. Use a new output directory for each run. A missing PMID or unavailable PDF is recorded; no PDF download is guaranteed by PubMed indexing alone.

The Excel script needs `@oai/artifact-tool`, supplied by a compatible document runtime. It is not installed by this repository's Python dependencies. Set `CODEX_NODE` to the actual Node executable and `CODEX_NODE_MODULES` to the available package location when needed. Do not assume the Windows desktop runtime exists in a Linux cloud VM. Missing Node/package support leaves JSON and CSV intact and makes the helper report XLSX export failure; complete XLSX with the cloud's spreadsheet capability before declaring delivery complete.

Verify exported XLSX row counts, unique table names, identifiers, dates, and all sheet previews. Verify downloaded PDFs by opening them and matching their titles. Optional `NCBI_EMAIL`, `NCBI_API_KEY`, and `OPENALEX_API_KEY` are environment configuration, not repository files.

## Using the workflow

Ask Codex:

> Use $pico-literature-search for community-dwelling older adults and translation/validation of GSEOH into Taiwanese Traditional Chinese. Research each database's search strategy, collect accessible PDFs, and export an Excel reading list. For unavailable full texts, retain the article title and identifiers, arrange acquisition using citation counts, and mark COSMIN pending until there is reliable full-text or external assessment evidence. Assess available measurement studies by property, with sources.

For a different PICO, research and write a new JSON profile with custom `queries`; pass `--config` and the corresponding PubMed query IDs. The helper does not translate arbitrary PICO text into a validated search strategy by itself.

```text
python .agents/skills/pico-literature-search/scripts/literature_tool.py rank --out outputs/cloud_smoke --top 20
```

`rank` looks up citation counts and preserves evidenced assessments; it does not generate COSMIN ratings. Unknown counts stay blank. Use `--quality-property` to compare existing evidenced ratings of the same property. No unsupported article-wide numerical quality score is calculated.

Cochrane and Scholar strategy generation and RIS import are supported. The helper does not bulk scrape Scholar or automate their website interfaces. The current official Codex Cloud documentation lists browser use as a limitation; use exports or a suitable browser-capable workflow and label any unexecuted sources.

Outputs include `manifest.json`, full abstracts and raw API responses, `文獻清單.csv`, `文獻清單.xlsx`, `搜尋式.html`, and downloaded PDFs. After `rank`, XLSX includes a missing-full-text priority sheet. Keep filled Excel notes before re-export because notes are not synced back to JSON. Save required task outputs explicitly; environment saved state is not durable source control or an archive.

The public GSEOH example contains bibliographic records only. It does not bundle personal correspondence, the Japanese attachment, local PDF paths, credentials, or study data. Supply the relevant translation permission and mother version separately in the task that needs them.
