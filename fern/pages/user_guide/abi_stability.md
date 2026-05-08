````markdown
# End-User Guide: cuVS ABI Stability

## What ABI Stability Means

ABI stability means that an application built with one compatible version of cuVS can continue to run with a newer compatible cuVS runtime, without the application needing to be rebuilt.

A simple way to think about it:

```text
Your application was built for a certain cuVS runtime family.
As long as you install a compatible runtime from that same family,
the application should continue to work.
```

ABI stands for **Application Binary Interface**. It is the runtime contract between a compiled application and the shared library it loads.

You usually do not need to understand the low-level details. The important point is this:

```text
ABI stability helps already-built applications keep working across compatible cuVS upgrades.
```

---

## Why ABI Stability Exists

Many applications use cuVS through another product, such as a database, search system, language binding, or packaged application.

Without ABI stability, every cuVS update could require those applications to be rebuilt or repackaged. That would make upgrades more difficult for both software vendors and users.

ABI stability allows a better model:

```text
Application vendor:
  Builds their application against a supported cuVS version.

End user:
  Installs a compatible cuVS runtime.

Result:
  The application can run without being rebuilt.
```

This makes cuVS easier to use in larger systems where applications and runtime libraries may be installed separately.

---

## Benefits for End Users

ABI stability provides several practical benefits.

### Easier upgrades

You can install compatible newer cuVS runtime versions without requiring every application to be rebuilt.

```text
Old model:
  Update cuVS -> rebuild or reinstall the application

ABI-stable model:
  Update cuVS within the same ABI family -> application keeps working
```

### Smaller application packages

Applications do not always need to bundle their own private copy of cuVS. Instead, they can rely on a compatible cuVS runtime installed on the system.

### More predictable compatibility

The ABI version tells you whether a cuVS runtime is expected to work with an already-built application.

### Safer deployments

A clear compatibility rule reduces surprises when updating runtime libraries.

---

## The Two Versions You May See

There are two related version numbers:

1. The **cuVS product version**
2. The **cuVS ABI version**

They answer different questions.

| Version | Example | What it means |
|---|---:|---|
| cuVS version | `26.04`, `26.06`, `26.08` | The normal cuVS release version |
| ABI version | `1.1`, `1.2`, `2.0` | The binary compatibility version |

The cuVS version tells you which release you are using.

The ABI version tells you whether a compiled application can run with that runtime.

---

## ABI Versioning Scheme

The ABI version uses this format:

```text
<abi-major>.<abi-minor>
```

For example:

```text
1.2
```

Where:

```text
1 = ABI major version
2 = ABI minor version
```

The ABI major version is the most important part for compatibility.

---

## ABI Major Version

The ABI major version identifies a compatibility family.

Examples:

```text
ABI 1.x
ABI 2.x
ABI 3.x
```

Applications built for one ABI major version are expected to run with runtimes from the same ABI major version, assuming the runtime is new enough.

For example:

```text
Application built for ABI 1.x
Runtime provides ABI 1.x
Compatible
```

But:

```text
Application built for ABI 1.x
Runtime provides ABI 2.x
Not guaranteed to be compatible
```

When the ABI major version changes, it means cuVS may have made changes that require applications to be rebuilt or updated.

---

## ABI Minor Version

The ABI minor version increases when compatible functionality is added.

For example:

```text
ABI 1.1 -> ABI 1.2
```

This means the runtime is still in the same ABI major compatibility family, but it may include additional compatible features.

A newer ABI minor version can usually support applications built with an older ABI minor version from the same ABI major family.

Example:

```text
Application built with ABI 1.1
Runtime provides ABI 1.2

Result:
  Compatible
```

But the reverse is not always true:

```text
Application built with ABI 1.2
Runtime provides ABI 1.1

Result:
  Not guaranteed to work
```

The runtime must be the same version or newer than what the application expects.

---

## Compatibility Rule

A cuVS runtime is compatible with an application when both of these are true:

1. The runtime has the same **ABI major version** as the version the application was built for.
2. The runtime is the same or newer than the version the application was built for.

In simple terms:

```text
Same ABI major version + same or newer runtime = compatible
```

---

## Example Compatibility Table

| Application Built With | Installed Runtime | Compatible? | Why |
|---|---:|---:|---|
| cuVS `26.04`, ABI `1.1` | cuVS `26.06`, ABI `1.2` | Yes | Same ABI major, newer runtime |
| cuVS `26.04`, ABI `1.1` | cuVS `26.04`, ABI `1.1` | Yes | Exact match |
| cuVS `26.04`, ABI `1.1` | cuVS `26.02`, ABI `1.0` | No | Runtime is older |
| cuVS `26.04`, ABI `1.1` | cuVS `26.08`, ABI `2.0` | No | ABI major changed |

---

## Shared Library Names

On Linux systems, you may see cuVS shared libraries with names like this:

```text
libcuvs_c.so.1.2
```

This name includes the ABI version:

```text
libcuvs_c.so.<abi-major>.<abi-minor>
```

For example:

```text
libcuvs_c.so.1.2
```

Means:

```text
ABI major = 1
ABI minor = 2
```

You may also see a shorter name:

```text
libcuvs_c.so.1
```

This refers to the ABI major version only.

In most cases, applications should use the ABI-major versioned library name, such as:

```text
libcuvs_c.so.1
```

That allows the system to use compatible newer ABI-minor versions automatically.

---

## What Happens When the ABI Major Version Changes?

When the ABI major version changes, cuVS has entered a new compatibility family.

For example:

```text
ABI 1.x -> ABI 2.x
```

This usually means applications built for the older ABI may need to be updated or rebuilt.

As an end user, this means:

```text
Do not assume an application built for ABI 1.x will work with only an ABI 2.x runtime.
```

The safest approach is to install the cuVS runtime version recommended by the application vendor or package manager.

---

## What End Users Should Do

Most users do not need to manually manage ABI details. Package managers and application vendors should normally handle this for you.

However, when installing or troubleshooting cuVS manually, follow these rules:

1. Check which cuVS version your application requires.
2. Install the same or newer compatible cuVS runtime.
3. Make sure the ABI major version matches.
4. Avoid replacing an ABI `1.x` runtime with only an ABI `2.x` runtime unless your application supports ABI `2.x`.
5. Prefer package-manager-managed installs when available.

Simple rule:

```text
Use the runtime version your application asks for,
or a newer runtime with the same ABI major version.
```

---

## Common Questions

### Can I update cuVS without rebuilding my application?

Yes, as long as the new cuVS runtime is ABI-compatible with the version your application was built against.

That usually means:

```text
Same ABI major version
Same or newer runtime version
```

---

### Can I use an older cuVS runtime with a newer application?

Usually no.

A newer application may depend on symbols or behavior that do not exist in the older runtime.

```text
Newer application + older runtime = not guaranteed to work
```

---

### Can I use ABI 2.x with an application built for ABI 1.x?

Not unless the application vendor says it supports ABI 2.x.

A change in ABI major version means compatibility is not guaranteed across that boundary.

---

### Why does the library name include a number?

The number helps the system load a compatible cuVS runtime.

For example:

```text
libcuvs_c.so.1
```

Means the application expects the ABI `1.x` compatibility family.

---

## Summary

ABI stability makes cuVS easier and safer to use in real applications.

It allows applications built against one supported cuVS version to keep working with compatible newer cuVS runtimes. This reduces rebuilds, simplifies upgrades, and gives users a clearer way to understand which runtime versions are safe to install.

The most important rule is:

```text
Use the same ABI major version,
and use a runtime that is the same version or newer than the one the application was built with.
```

For example:

```text
Built with ABI 1.1
Runs with ABI 1.2
Does not necessarily run with ABI 2.0
```

When in doubt, use the cuVS runtime version recommended by your application vendor or package manager.
````
