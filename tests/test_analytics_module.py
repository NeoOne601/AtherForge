from types import SimpleNamespace

from src.config import get_settings
from src.modules.analytics.module import AnalyticsModule


def test_analytics_module_generates_report_artifacts_from_livefolder_csv():
    settings = get_settings()
    csv_path = settings.live_folder / "sales.csv"
    csv_path.write_text("month,revenue\nJan,10\nFeb,20\nMar,15\n", encoding="utf-8")

    module = AnalyticsModule()
    state = SimpleNamespace(
        settings=settings,
        export_engine=SimpleNamespace(_markdown_to_pdf=lambda text: b"%PDF-test%"),
        sparse_index=None,
    )

    result = module.execute_tool(
        "analyze_data",
        {
            "source": "sales.csv",
            "question": "Summarize this file and create exports.",
            "format": "both",
            "include_visual": False,
        },
        state=state,
    )

    assert isinstance(result, dict)
    assert "sales.csv" in result["content"]
    assert any(name.endswith(".md") for name in result["attachments"])
    assert any(name.endswith(".pdf") for name in result["attachments"])
    assert result["citations"][0]["source"] == "sales.csv"

    for attachment in result["attachments"]:
        assert (settings.generated_dir / attachment).exists()
