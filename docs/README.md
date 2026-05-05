# Building Documentation

The cuVS documentation is a Fern project in [../fern](../fern).

## Preview locally

Install the Fern CLI and run the local preview from the repository root:

```bash
npm install -g fern-api
fern docs dev
```

Fern serves the preview at [http://localhost:3000](http://localhost:3000) by default.

## Validate

```bash
fern check --warnings --strict-broken-links
fern docs md check
```
