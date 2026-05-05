# cuVS Fern documentation

The cuVS documentation lives in this Fern project. Pages are in `fern/pages`, and the sidebar navigation is configured in `fern/docs.yml`.

## Preview locally

Start the local preview server from the repository root:

```bash
fern/build_docs.sh dev
```

Fern serves the local preview at `http://localhost:3000` by default.

## Validate

Run Fern's checks before publishing changes:

```bash
fern/build_docs.sh check
```

## Publish

Create a Fern preview deployment:

```bash
fern/build_docs.sh preview
```

Publish to the instance configured in `fern/docs.yml` with:

```bash
fern/build_docs.sh publish
```
