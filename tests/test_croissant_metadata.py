import json
import re
from collections import Counter
from pathlib import Path


def test_croissant_metadata_is_1_1_and_internally_consistent():
    metadata = json.loads((Path(__file__).parents[1] / "croissant.json").read_text())

    ids, refs = [], []

    def walk(value):
        if isinstance(value, dict):
            if "@id" in value:
                (ids if len(value) > 1 else refs).append(value["@id"])
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(metadata)

    assert metadata["conformsTo"] == "http://mlcommons.org/croissant/1.1"
    assert {
        "rai:dataLimitations",
        "rai:dataBiases",
        "rai:personalSensitiveInformation",
        "rai:dataUseCases",
        "rai:dataSocialImpact",
        "rai:hasSyntheticData",
    } <= set(metadata)
    assert metadata["rai:hasSyntheticData"] is True
    assert not [value for value, count in Counter(ids).items() if count > 1]
    assert not (set(refs) - set(ids))
    assert any(record_set["@id"] == "split_files" for record_set in metadata["recordSet"])
    assert not any(record_set["@id"] == "split_membership" for record_set in metadata["recordSet"])
    assert all(
        re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
        for item in metadata["distribution"]
        if item["@id"].startswith(("dataset-archive", "baseline-results"))
    )
