"""Quick validation script for all backend modules."""
import json

print("=" * 60)
print("  FarmWise Backend Module Validation")
print("=" * 60)

# 1. Config
from backend import config
print("\n[1] Config")
print(f"    Model: {config.GROQ_MODEL}")
print(f"    API Key set: {bool(config.GROQ_API_KEY)}")
print(f"    Dataset: {config.DATASET_PATH}")

# 2. Dataset
from backend.dataset import load_dataset, get_schema_summary, get_column_names, get_unique_values
df = load_dataset()
print(f"\n[2] Dataset: {df.shape[0]} rows x {df.shape[1]} cols")
print(f"    Columns: {get_column_names()}")

# 3. Tools
from backend.tools import (
    filter_by_country, filter_by_item, get_yield_trend,
    compare_countries, get_top_countries, get_correlation,
    get_summary_statistics, get_pesticide_usage, get_tools_description,
    TOOL_SCHEMAS, dispatch_tool_call,
)

print(f"\n[3] Tools: {len(TOOL_SCHEMAS)} schemas registered")

r = filter_by_country("India")
print(f"    filter_by_country('India'): {len(r)} rows")

r = filter_by_item("Wheat")
print(f"    filter_by_item('Wheat'): {len(r)} rows")

r = get_yield_trend("India", "Wheat")
print(f"    get_yield_trend('India', 'Wheat'): {len(r)} rows")

r = compare_countries(["India", "China"], "Wheat")
print(f"    compare_countries(['India','China'], 'Wheat'): {len(r)} rows")

r = get_top_countries(metric="hg/ha_yield", item="Wheat", year=2013, top_n=3)
print(f"    get_top_countries('Wheat', 2013, 3): {[(x['Area'], x['hg/ha_yield']) for x in r]}")

r = get_correlation("average_rain_fall_mm_per_year", "hg/ha_yield")
print(f"    get_correlation(rain, yield): r={r['correlation']}, points={r['data_points']}")

r = get_summary_statistics("hg/ha_yield")
print(f"    get_summary_statistics('hg/ha_yield'): mean={r['summary']['mean']}, median={r['summary']['median']}")

r = get_pesticide_usage("India")
print(f"    get_pesticide_usage('India'): {len(r)} rows")

# Test dispatcher
r = dispatch_tool_call("get_top_countries", {"metric": "hg/ha_yield", "item": "Maize", "top_n": 3})
print(f"    dispatch_tool_call('get_top_countries', Maize): {len(r)} rows")

print(f"\n    Tools description preview:\n    {get_tools_description()[:200]}...")

# 4. Chart generator
from backend.chart_generator import create_chart, figure_to_json
data = get_top_countries(metric="hg/ha_yield", item="Wheat", year=2013, top_n=5)
fig = create_chart("bar", data, x="Area", y="hg/ha_yield", title="Top 5 Wheat Producers 2013")
chart_json = figure_to_json(fig)
print(f"\n[4] Chart generator: created bar chart, JSON length={len(chart_json)}")

# 5. Insight engine
from backend.insight_engine import generate_insight, format_data_for_display
data = get_yield_trend("India", "Wheat")
insights = generate_insight(data, y_column="hg/ha_yield")
print(f"\n[5] Insight engine:")
print(f"    Rows: {insights['row_count']}")
print(f"    Average: {insights['average']}")
print(f"    Trend: {insights['trend']}")
print(f"    Growth rate: {insights['growth_rate_pct']}%")
print(f"    Outliers: {insights['outlier_count']}")

display = format_data_for_display(data, max_rows=3)
print(f"    format_data_for_display: {len(display)} rows")

# 6. Memory
from backend.memory import get_memory, delete_memory
mem = get_memory("test-session")
mem.add_message(role="system", content="You are a test bot.")
mem.add_message(role="user", content="Hello")
mem.add_message(role="assistant", content="Hi there!")
msgs = mem.get_messages()
print(f"\n[6] Memory: {len(msgs)} messages stored")
delete_memory("test-session")
print("    Session deleted OK")

# 7. Prompts
from backend.prompts import build_system_prompt
prompt = build_system_prompt(
    schema=get_schema_summary(),
    sample_rows=[{"Area": "India", "Item": "Wheat", "Year": 2013}],
    tools_description="test tools",
    countries=["India", "China"],
    crops=["Wheat", "Rice"],
    year_range=(1990, 2013),
)
print(f"\n[7] Prompts: system prompt length={len(prompt)} chars")

print("\n" + "=" * 60)
print("  ALL MODULES VALIDATED SUCCESSFULLY")
print("=" * 60)
