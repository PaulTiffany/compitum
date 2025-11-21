# %% [markdown]
# Materials Project Manifold Audit (Simulated Routing)
#
# Requires an MP API key. This demo computes SRMF proxies, a curvature proxy,
# and a Lyapunov leak proxy, then ranks candidates. Set MP_API_KEY in the environment.

# %%
import os
from compitum.integrations.materials_project_audit import audit_the_manifold

MP_API_KEY = os.environ.get("MP_API_KEY", "")
if not MP_API_KEY:
    print("MP_API_KEY not set. Set it to run the live query against Materials Project.")
else:
    criteria = {"elements": ["La","Ni","O"], "nelements": 3}
    df = audit_the_manifold(MP_API_KEY, criteria)
    display(df.sort_values("curvature_kappa", ascending=False).head(25))

# %%
if MP_API_KEY:
    ax = df.plot.scatter(x="curvature_kappa", y="stability_leak", title="Manifold Audit")
    display(ax.figure)