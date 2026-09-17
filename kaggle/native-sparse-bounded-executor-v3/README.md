# Phase 6G bounded executor v3 — exact IQP cache integration

Version 2 proved the explicit 2,000,000,000-byte cache can run the real model
at about 3.5 GiB RSS, but it disabled IQP and therefore changed numerical
behavior/routes. Version 3 keeps IQP enabled and adds one narrow source-pointer
hook inside `iqp.cpp`: the selected expert plane is resolved to the prepared
cache slot before panel decode. The control arm continues to use the normal
source pointer. This isolates bounded storage from arithmetic semantics.

The model, native top-8 routing, cache policy/capacity, 64-token workload, and
three repetitions are unchanged.
