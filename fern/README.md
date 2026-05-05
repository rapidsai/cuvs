# cuVS Fern documentation

The cuVS documentation lives in this Fern project. Pages are in `fern/pages`, and the sidebar navigation is configured in `fern/docs.yml`.

## Preview locally

Install the Fern CLI and start the docs server:

```bash
npm install -g fern-api
fern docs dev
```

Fern serves the local preview at `http://localhost:3000` by default.

## Validate

Run Fern's checks before publishing changes:

```bash
fern check --warnings --strict-broken-links
fern docs md check
```

## Publish

Publish to the instance configured in `fern/docs.yml` with:

```bash
fern generate --docs
```
