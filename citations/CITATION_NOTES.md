# Citation Notes

- Add dataset citation and source URL here.
- Track versioned software citations (PyMC, ArviZ, scikit-learn).

Verification log (Crossref API)
Date: 2026-01-16

DOI | Status
--- | ---
10.1093/biomet/63.3.581 | 200
10.1214/06-BA117A | 200
10.1016/j.jmva.2009.04.008 | 200
10.1007/s11222-016-9696-4 | 200
10.1214/20-BA1221 | 200
10.1198/016214506000001437 | 200
10.1002/widm.39 | 200
10.1098/rsta.2015.0202 | 200
10.1214/09-AOS735 | 200
10.1214/17-BA1091 | 200

Verification command (requires curl and jq):
```bash
# Check a single DOI via Crossref API
curl -s "https://api.crossref.org/works/10.1093/biomet/63.3.581" | jq '.status'
```
