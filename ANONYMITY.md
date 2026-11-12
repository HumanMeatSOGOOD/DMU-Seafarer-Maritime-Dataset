# Anonymity notes

This package is prepared for **double-anonymized review**. It intentionally
contains no author names, affiliations, e-mail addresses, funding information,
or acknowledgments.

## If you upload it to GitHub

A public repository can still reveal authorship through:

- the GitHub account owner and profile,
- the commit author name/e-mail,
- the account's other repositories,
- the repository URL if it contains a name or institution,
- README badges, funding notes, or acknowledgments.

For review, use one of these options:

1. **Anonymous repository mirror** (recommended): upload the folder to
   anonymous.4open.science or a comparable service that strips account
   identity.
2. **Anonymous GitHub account**: create a dedicated account with no personal
   profile information and a neutral repository name; set a neutral commit
   identity before committing:

   ```bash
   git config user.name "Anonymous"
   git config user.email "anonymous@example.com"
   ```

3. **Private repository with editor/reviewer access**, if the journal supports
   it.

Do not use a personal GitHub account for the reviewer-facing link if the
manuscript is under double-anonymized review.

## What is intentionally absent

- Author names, affiliations, e-mails, ORCID iDs.
- Acknowledgments and funding statements.
- Personal records, seafarer names, vessel names, IMO numbers, timestamps.
- Manuscript PDFs, cover letters, title pages, or author biographies.
- Any `.git` history (the folder is delivered without a repository history).

## After acceptance

Update the following with the final author information:

- `LICENSE` (copyright holder)
- `DATA_LICENSE.md` (attribution)
- `README.md` (repository URL and contact)
- `CITATION.cff`, if you add one

Then publish the repository from the authors' institutional account or a
stable lab account, and replace the anonymous review link in the manuscript
with the final URL.
