# Building Documentation

The cuVS documentation is a Fern project in [../fern](../fern).

## Preview locally

Install the Fern CLI and run the local preview from the repository root:

```bash
fern/build_docs.sh dev
```

Fern serves the preview at [http://localhost:3000](http://localhost:3000) by default.

## Validate

```bash
fern/build_docs.sh check
```

## Build

Create a Fern preview deployment:

```bash
fern/build_docs.sh preview
```

Publish the production docs site:

```bash
fern/build_docs.sh publish
```
