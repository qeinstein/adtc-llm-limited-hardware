# ik_llama.cpp compatibility probe v3

Version 1 mishandled the CLI help exit status; version 2 then assumed the
mainline `-ngl` alias, which this ik_llama CLI does not expose. Version 3
uses the actual CPU-only option surface, runs the exact model smoke, and runs
the 64-token/3-repeat resident benchmark. It records runtime-repack support
(`-rtr`) for later donor comparison without changing model weights or routing.
