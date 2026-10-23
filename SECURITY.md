# Security policy

## Supported versions

Security fixes go into the latest release.

## Reporting a vulnerability

Report privately through the repository's **Security** tab ("Report a vulnerability").
Do not open a public issue.

Include the strdcheck version, your OS, what goes wrong, and how to reproduce it. Expect an
acknowledgement within a week.

## Threat model

strdcheck is a local numerical-accuracy library, pytest plugin and command-line tool.

- **No credentials.** strdcheck needs no API keys and makes no calls to model or analytics
  services.
- **One network host.** The downloader fetches the NIST StRD files over HTTPS from
  `https://www.itl.nist.gov/div898/strd/` and refuses any other URL. Every file is checked
  against the SHA-256 recorded in the package manifest before it is written to the cache
  (atomic rename); a mismatch is an error, never a warning.
- **Downloaded data is untrusted input.** The parsers read ASCII text only, validate the
  layout and the number of observations against the header, and never execute or evaluate
  file content.
- **Cache and run directories** hold the StRD files and intermediate results only. They are
  local, gitignored, and safe to delete.
