"""Which client an incoming request belongs to, for rate limiting (PLAN.md Phase 5 step 7).

Behind a reverse proxy (Render's), every request arrives from the proxy's own address, so one
client's limit would be shared by everyone. The proxy appends the address it saw to the
``X-Forwarded-For`` list. A client can put anything *before* that entry, so only the entry added by
the trusted proxy(ies) may be believed: the one ``trusted_proxy_hops`` places from the **right**.

* ``trusted_proxy_hops = 0`` (the default): ignore ``X-Forwarded-For`` completely and use the
  connection's address. Right for local use and for a deployment without a proxy.
* ``trusted_proxy_hops = 1``: one proxy in front, the last entry is the real client.

If the header is shorter than the number of hops, the connection's address is used. **[Unverified]**
how many hops Render's proxy adds; this must be checked against Render's current documentation
before deploying (PLAN.md Phase 5 step 7). Nothing here is tied to a particular proxy.
"""

from __future__ import annotations

from collections.abc import Callable

from starlette.requests import Request


def client_address(request: Request, trusted_proxy_hops: int) -> str:
    """The address to rate-limit this request under."""
    peer = request.client.host if request.client else "unknown"
    if trusted_proxy_hops <= 0:
        return peer
    forwarded = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",")]
    forwarded = [p for p in forwarded if p]
    if len(forwarded) < trusted_proxy_hops:
        return peer
    return forwarded[-trusted_proxy_hops]


def key_function(trusted_proxy_hops: int) -> Callable[[Request], str]:
    return lambda request: client_address(request, trusted_proxy_hops)
