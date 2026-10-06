# AcademicHelper

## PICO literature search for Codex Cloud

The repository includes a Codex skill at `.agents/skills/pico-literature-search/` for source-grounded PubMed, Cochrane Library, and Google Scholar search strategies, open-access PDF retrieval, and an Excel reading list. PubMed metadata and PMC Cloud PDFs use the standard-library Python helper; Cochrane and Scholar use platform searches and RIS imports.

Missing full texts remain in a priority list with DOI-matched OpenAlex citation counts and explicit COSMIN assessment status. Citation counts do not substitute for research quality; COSMIN ratings require evidence for the same measurement property.

See [Codex Cloud setup and validation](docs/codex-cloud-literature.md) for dependencies, network access, and a bounded first run. Cloud execution and XLSX dependencies must be verified in the selected environment before claiming the workflow is ready.
