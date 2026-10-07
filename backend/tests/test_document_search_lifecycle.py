import json
import urllib.error
import urllib.request


BASE_URL = "http://127.0.0.1:8000"

CREATE_TOKEN = "zephyrlattice"
UPDATE_TOKEN = "quartznebula"


def request(method, path, body=None):
    data = None
    headers = {}

    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(
        BASE_URL + path,
        data=data,
        headers=headers,
        method=method,
    )

    try:
        with urllib.request.urlopen(req) as response:
            return (
                response.status,
                json.loads(response.read().decode("utf-8")),
            )
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8")
        raise RuntimeError(
            f"{method} {path} failed: "
            f"{exc.code} {payload}"
        ) from exc


def lexical_search(query):
    status, payload = request(
        "POST",
        "/search/lexical",
        {
            "query": query,
            "corpus": "application",
            "top_k": 10,
        },
    )
    assert status == 200
    return payload


def contains_document(payload, document_id):
    return any(
        item["document_id"] == document_id
        for item in payload["results"]
    )


document_id = None

try:
    # --------------------------------------------------
    # 1. CREATE
    # --------------------------------------------------
    print("\n[1] CREATE")

    status, created = request(
        "POST",
        "/documents",
        {
            "title": "Lifecycle Retrieval Test",
            "content": (
                "The zephyrlattice protocol is a unique "
                "sentinel phrase used for retrieval testing."
            ),
            "author": "Integration Test",
            "source": "automated-test",
            "publication_year": 2026,
        },
    )

    assert status in (200, 201)
    document_id = created["document_id"]

    print("document_id:", document_id)
    print("PASS: document created")

    # --------------------------------------------------
    # 2. SEARCH AFTER CREATE
    # --------------------------------------------------
    print("\n[2] SEARCH AFTER CREATE")

    payload = lexical_search(CREATE_TOKEN)

    assert contains_document(payload, document_id)
    print("PASS: created document is searchable")

    # --------------------------------------------------
    # 3. UPDATE
    # --------------------------------------------------
    print("\n[3] UPDATE")

    status, updated = request(
        "PUT",
        f"/documents/{document_id}",
        {
            "content": (
                "The quartznebula protocol replaces the "
                "previous sentinel phrase after an update."
            )
        },
    )

    assert status == 200
    print("PASS: document updated")

    # --------------------------------------------------
    # 4. OLD TERM MUST DISAPPEAR
    # --------------------------------------------------
    print("\n[4] VERIFY OLD TERM DISAPPEARS")

    old_payload = lexical_search(CREATE_TOKEN)

    assert not contains_document(
        old_payload,
        document_id,
    )

    print("PASS: old content is no longer searchable")

    # --------------------------------------------------
    # 5. NEW TERM MUST APPEAR
    # --------------------------------------------------
    print("\n[5] VERIFY NEW TERM APPEARS")

    new_payload = lexical_search(UPDATE_TOKEN)

    assert contains_document(
        new_payload,
        document_id,
    )

    print("PASS: updated content is searchable")

    # --------------------------------------------------
    # 6. VECTOR SEARCH
    # --------------------------------------------------
    print("\n[6] VECTOR SEARCH")

    status, vector_payload = request(
        "POST",
        "/search/vector",
        {
            "query": (
                "quartznebula protocol sentinel phrase"
            ),
            "corpus": "application",
            "top_k": 10,
            "ef_search": 40,
        },
    )

    assert status == 200
    assert contains_document(
        vector_payload,
        document_id,
    )

    print("PASS: updated embedding is searchable")

    # --------------------------------------------------
    # 7. HYBRID SEARCH
    # --------------------------------------------------
    print("\n[7] HYBRID SEARCH")

    status, hybrid_payload = request(
        "POST",
        "/search/hybrid",
        {
            "query": UPDATE_TOKEN,
            "corpus": "application",
            "top_k": 10,
            "candidate_k": 10,
            "ef_search": 40,
            "rrf_k": 60,
        },
    )

    assert status == 200
    assert contains_document(
        hybrid_payload,
        document_id,
    )

    print("PASS: hybrid retrieval finds document")

    # --------------------------------------------------
    # 8. DELETE
    # --------------------------------------------------
    print("\n[8] DELETE")

    status, _ = request(
        "DELETE",
        f"/documents/{document_id}",
    )

    assert status in (200, 204)
    print("PASS: document deleted")

    # Prevent finally from deleting twice.
    deleted_id = document_id
    document_id = None

    # --------------------------------------------------
    # 9. SEARCH AFTER DELETE
    # --------------------------------------------------
    print("\n[9] VERIFY DELETE")

    payload = lexical_search(UPDATE_TOKEN)

    assert not contains_document(
        payload,
        deleted_id,
    )

    print("PASS: deleted document is no longer searchable")

    print("\n====================================")
    print("DOCUMENT SEARCH LIFECYCLE — PASS")
    print("====================================")

finally:
    # Cleanup if an earlier assertion fails.
    if document_id is not None:
        try:
            request(
                "DELETE",
                f"/documents/{document_id}",
            )
            print(
                f"\nCleanup: deleted document {document_id}"
            )
        except Exception as exc:
            print(
                f"\nCleanup warning: {exc}"
            )
