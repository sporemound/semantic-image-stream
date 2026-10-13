import csv
import io
import json

import httpx
import pytest

from semantic_image_stream.training_data import PublicDataClient, checked_url, story_examples


@pytest.mark.parametrize("url", ["http://huggingface.co/example", "https://user:secret@huggingface.co/x",
                                  "https://huggingface.co:1234/x", "https://huggingface.co.evil.example/x",
                                  "https://127.0.0.1/private", "file:///private"])
def test_training_download_rejects_unapproved_destinations(url):
    with pytest.raises(ValueError):
        checked_url(url)


def test_training_download_rechecks_redirect_and_caps_response():
    calls = []
    def handler(request):
        calls.append(request.url)
        return httpx.Response(302, headers={"Location": "https://evil.example/data"})
    client = PublicDataClient()
    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(ValueError):
            client.get("https://huggingface.co/public", 1024)
        assert len(calls) == 1
        client.client.close()
        client.client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"x" * 100)))
        with pytest.raises(ValueError):
            client.get("https://huggingface.co/public", 10)
    finally:
        client.close()


def test_story_pack_keeps_only_explicit_supported_train_answers():
    columns = ["story_name", "story_section", "question", "answer1", "attribute", "ex_or_im"]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    base = dict(story_name="test-story", story_section="The fox walked beside the lake and saw a wolf.",
                question="Where did the fox walk?", answer1="beside the lake", attribute="setting", ex_or_im="explicit")
    for changes in ({}, {"answer1": "on the moon"}, {"ex_or_im": "implicit"}, {"attribute": "prediction"}):
        writer.writerow({**base, **changes})
    pack = story_examples(output.getvalue().encode())
    assert len(pack["examples"]) == 1
    row = pack["examples"][0]
    assert row["source"] == base["story_section"] and row["answer"] == base["answer1"]
    assert row["provenance"]["split"] == "train"
    assert json.loads(json.dumps(pack)) == pack
