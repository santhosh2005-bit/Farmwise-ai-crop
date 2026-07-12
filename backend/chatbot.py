"""
chatbot.py — Orchestrator that ties together the Groq LLM, tools,
memory, and insight engine.

This is the **only** module that talks to the Groq API.  It:

1. Builds the system prompt (with live dataset context).
2. Sends the conversation to the LLM with tool schemas.
3. Handles tool-call responses in a loop until the LLM emits a final
   answer.
4. Returns a structured response including text and optional chart spec.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import pandas as pd
from groq import AsyncGroq

from backend import config, dataset
from backend.chart_generator import create_chart, figure_to_json
from backend.insight_engine import format_data_for_display, generate_insight
from backend.memory import ConversationMemory, get_memory, delete_memory
from backend.prompts import build_system_prompt
from backend.tools import (
    TOOL_SCHEMAS,
    dispatch_tool_call,
    get_tools_description,
)

logger = logging.getLogger(__name__)

# Maximum number of tool-call round-trips before forcing a final answer
_MAX_TOOL_ITERATIONS: int = 5
# Maximum total individual tool calls across all iterations
_MAX_TOTAL_TOOL_CALLS: int = 3


class Chatbot:
    """Stateful chatbot backed by Groq (llama-3.3-70b-versatile).

    Parameters
    ----------
    session_id : str
        Unique identifier for the conversation session.
    """

    def __init__(
        self,
        session_id: str,
        custom_api_key: str | None = None,
        model_choice: str | None = None,
    ) -> None:
        self.session_id: str = session_id
        resolved_key = custom_api_key.strip() if (custom_api_key and custom_api_key.strip()) else config.GROQ_API_KEY
        self.client: AsyncGroq = AsyncGroq(api_key=resolved_key, max_retries=2)
        self.model: str = model_choice if (model_choice and model_choice.strip()) else config.GROQ_MODEL
        self.memory: ConversationMemory = get_memory(session_id)

        # Inject the system prompt on first use
        if not self.memory.messages:
            system_prompt = self._build_system_prompt()
            self.memory.add_message(role="system", content=system_prompt)

    # ── Public API ───────────────────────────────────────────

    async def ask(self, user_message: str) -> dict[str, Any]:
        """Process a user message and return a structured response.

        Parameters
        ----------
        user_message : str
            The user's natural-language question.

        Returns
        -------
        dict[str, Any]
            Keys:
            - ``"answer"`` (str): The LLM's textual response.
            - ``"chart_json"`` (str | None): Plotly figure JSON, or *None*.
            - ``"data"`` (list[dict] | None): Supporting tabular data.
            - ``"suggestions"`` (list[str] | None): Follow-up questions.
            - ``"insights"`` (dict | None): Automatic data insights.
        """
        # Add user message to memory
        self.memory.add_message(role="user", content=user_message)

        # Run the tool-calling loop
        result = await self._run_tool_loop()

        return result

    def reset(self) -> None:
        """Clear conversation history for this session."""
        delete_memory(self.session_id)

    # ── Internal: system prompt builder ──────────────────────

    def _build_system_prompt(self) -> str:
        """Build the system prompt with live dataset context."""
        df = dataset.get_dataframe()
        return build_system_prompt(
            schema=dataset.get_schema_summary(),
            sample_rows=[],
            tools_description=get_tools_description(),
            countries=[],
            crops=[],
            year_range=(int(df["Year"].min()), int(df["Year"].max())),
        )

    def _get_safe_messages(self, all_msgs: list[dict[str, Any]], target_count: int = 8) -> list[dict[str, Any]]:
        """Slice the messages list safely so we never orphan a tool message.

        Always preserves the system prompt (at index 0) and keeps a slice of the
        remaining messages starting at a 'user' message boundary.
        """
        if len(all_msgs) <= 1:
            return all_msgs

        system_msg = all_msgs[0]
        non_system = all_msgs[1:]

        # If the non-system list is already within target_count, keep it all
        if len(non_system) <= target_count:
            return all_msgs

        # Find all indices in non_system where the role is 'user'
        user_indices = [i for i, msg in enumerate(non_system) if msg.get("role") == "user"]

        if not user_indices:
            # Fallback to simple slicing if no user message is found
            return [system_msg] + non_system[-target_count:]

        # Find the oldest 'user' message index that doesn't exceed target_count,
        # or at least keeps the slice size under target_count as much as possible.
        start_idx = user_indices[0]
        for idx in user_indices:
            remaining = len(non_system) - idx
            if remaining <= target_count:
                start_idx = idx
                break
        else:
            # If all user messages lead to remaining counts > target_count,
            # use the last user message to avoid orphaning tool calls in the active turn.
            start_idx = user_indices[-1]

        return [system_msg] + non_system[start_idx:]

    # ── Internal: Groq tool-calling loop ─────────────────────

    async def _run_tool_loop(self) -> dict[str, Any]:
        """Execute the tool-calling loop until the LLM produces a final answer.

        Returns
        -------
        dict[str, Any]
            Structured response with answer, charts, reasoning, data, suggestions, insights.
        """
        tool_data: list[dict[str, Any]] = []  # accumulate data from tools
        charts_json: list[str] = []
        reasoning_steps: list[dict[str, Any]] = []
        iterations = 0
        total_tool_calls = 0  # hard limit on individual tool calls

        while iterations < _MAX_TOOL_ITERATIONS:
            iterations += 1
            # Only send the system prompt + the last 8 conversation turns to stay under token limits
            all_msgs = self.memory.get_messages()
            messages = self._get_safe_messages(all_msgs, target_count=8)

            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    tool_choice="auto",
                    temperature=0.3,
                    max_tokens=800,
                )
            except Exception as e:
                logger.error(f"Groq API error: {e}")
                err_str = str(e).lower()
                if "rate limit" in err_str or "429" in err_str:
                    friendly_error = "I am currently receiving too many requests and hit my rate limit. Please wait a moment, or switch to a different AI model in the settings."
                elif "context" in err_str or "length" in err_str:
                    friendly_error = "This conversation has gotten too long for me to process. Please refresh the page or clear the chat history."
                else:
                    friendly_error = f"I encountered an unexpected error communicating with the AI service: {str(e)}"

                return {
                    "answer": f"**API Error:** {friendly_error}",
                    "chart_json": None,
                    "charts_json": [],
                    "data": None,
                    "suggestions": None,
                    "insights": None,
                    "reasoning": reasoning_steps,
                }

            choice = response.choices[0]
            message = choice.message

            # Record thoughts from the model if they exist
            if message.content:
                reasoning_steps.append({"type": "thought", "content": message.content})

            # ── Case 1: LLM wants to call tool(s) ───────────
            if message.tool_calls:
                # Store the assistant message (with tool_calls) in memory
                tool_calls_raw = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ]
                self.memory.add_message(
                    role="assistant",
                    content=message.content,
                    tool_calls=tool_calls_raw,
                )

                # Execute each tool call — enforce hard limit, reuse results for identical calls to prevent redundant executions
                executed_results = {}  # cache results of executed functions in this turn

                for tc in message.tool_calls:
                    if total_tool_calls >= _MAX_TOTAL_TOOL_CALLS:
                        logger.warning("Max total tool calls reached — stopping early.")
                        # Inject a fake tool result so memory stays valid
                        self.memory.add_message(
                            role="tool",
                            content=json.dumps({"note": "Tool call limit reached. Please generate the final answer now using data already retrieved."}),
                            tool_call_id=tc.id,
                            name=tc.function.name,
                        )
                        continue

                    func_name = tc.function.name
                    func_args_str = tc.function.arguments
                    
                    # Reuse already executed identical tool calls in the same turn
                    call_key = (func_name, func_args_str)
                    if call_key in executed_results:
                        logger.info(f"Reusing cached result for duplicate tool call: {func_name}({func_args_str})")
                        result = executed_results[call_key]
                    else:
                        total_tool_calls += 1
                        try:
                            func_args = json.loads(func_args_str)
                        except json.JSONDecodeError:
                            func_args = {}

                        logger.info(f"Tool call: {func_name}({func_args})")
                        reasoning_steps.append({
                            "type": "tool_call",
                            "name": func_name,
                            "arguments": func_args
                        })

                        try:
                            result = dispatch_tool_call(func_name, func_args)
                        except Exception as e:
                            logger.error(f"Tool execution error: {e}")
                            result = {"error": str(e)}
                        
                        executed_results[call_key] = result

                    # Accumulate data — merge lists across multiple tool calls
                    flat_result = result
                    if isinstance(result, list) and len(result) == 1 and isinstance(result[0], dict) and "data" in result[0]:
                        flat_result = result[0]["data"]
                    elif isinstance(result, dict) and "data" in result:
                        flat_result = result["data"]

                    if isinstance(flat_result, list) and flat_result:
                        if not isinstance(tool_data, list):
                            tool_data = []
                        for row in flat_result:
                            if row not in tool_data:
                                tool_data.append(row)
                    # For aggregate/stats results (dict without 'data'), store the
                    # tool args so we can fall back to the full dataset for charts
                    elif isinstance(result, dict) and "error" not in result and "data" not in result:
                        # Remember which column was queried for histogram fallback
                        if not hasattr(self, "_last_stats_args"):
                            self._last_stats_args = {}
                        self._last_stats_args.update(func_args)

                    # Record tool result summary
                    row_count = 0
                    warning_msg = None
                    if isinstance(result, list):
                        row_count = len(result)
                        # compare_countries wraps result in [{"data": [...], "warning": ...}]
                        if len(result) == 1 and isinstance(result[0], dict):
                            if "data" in result[0]:
                                row_count = len(result[0]["data"])
                            if "warning" in result[0]:
                                warning_msg = result[0]["warning"]
                    elif isinstance(result, dict) and "data" in result:
                        row_count = len(result["data"])
                        if "warning" in result:
                            warning_msg = result["warning"]

                    if row_count > 0:
                        summary = f"Successfully retrieved {row_count} data records."
                    else:
                        summary = "No matching data found for the given parameters."

                    if warning_msg:
                        summary += f" ⚠️ WARNING: {warning_msg}"

                    if isinstance(result, dict) and "error" in result:
                        summary = f"Error: {result['error']}"

                    reasoning_steps.append({
                        "type": "tool_result",
                        "name": func_name,
                        "summary": summary
                    })

                    # Serialise result and add as tool message
                    result_str = json.dumps(result, default=str)

                    # Truncate very large results to stay within token limits
                    if len(result_str) > 800:
                        try:
                            # Try to keep it as valid JSON if it's a list
                            if isinstance(result, list):
                                if len(result) == 1 and isinstance(result[0], dict) and "data" in result[0]:
                                    # Copy to avoid modifying original inplace, truncate nested data
                                    # IMPORTANT: always preserve the 'warning' key so the LLM knows about missing countries
                                    truncated_dict = result[0].copy()
                                    truncated_dict["data"] = truncated_dict["data"][:5]
                                    result_str = json.dumps([truncated_dict], default=str) + "\n...[truncated, showing 5 rows]"
                                else:
                                    result_str = json.dumps(result[:5], default=str) + "\n...[truncated, showing 5 rows]"
                            elif isinstance(result, dict) and "data" in result:
                                result_tmp = result.copy()
                                result_tmp["data"] = result_tmp["data"][:5]
                                # Always preserve the warning key
                                if "warning" in result:
                                    result_tmp["warning"] = result["warning"]
                                result_str = json.dumps(result_tmp, default=str) + "\n...[truncated, showing 5 rows]"
                            else:
                                result_str = result_str[:800] + "...[truncated]"
                        except Exception:
                            result_str = result_str[:800] + "...[truncated]"

                    self.memory.add_message(
                        role="tool",
                        content=result_str,
                        tool_call_id=tc.id,
                        name=func_name,
                    )

                # Continue the loop — the LLM will process tool results
                continue

            # ── Case 2: LLM produced a final answer ──────────
            answer_text = message.content or "I couldn't generate a response."
            # Store the final assistant message in memory
            self.memory.add_message(role="assistant", content=answer_text)

            # Parse chart specs from the answer if present
            charts_json = self._extract_and_build_charts(answer_text, tool_data)

            # Extract data array if the tool wrapped it in a dictionary (e.g. compare_countries with warnings)
            actual_data = tool_data
            if isinstance(tool_data, list) and len(tool_data) == 1 and isinstance(tool_data[0], dict) and "data" in tool_data[0]:
                actual_data = tool_data[0]["data"]

            # ── AUTO-CHART FALLBACK ────────────────────────────────────────
            # If the model didn't output a chart spec, generate one automatically
            # based on which tool was called, so visualizations always appear.
            if not charts_json and actual_data and isinstance(actual_data, list) and len(actual_data) > 0:
                charts_json = self._auto_generate_chart(reasoning_steps, actual_data, answer_text)

            chart_json = charts_json[0] if charts_json else None

            # Extract follow-up suggestions
            suggestions = self._extract_suggestions(answer_text)

            # Generate automatic insights from tool data
            insights = None
            if tool_data:
                insights = generate_insight(tool_data)

            # Format data for display
            display_data = None
            if tool_data:
                display_data = format_data_for_display(tool_data)

            return {
                "answer": answer_text,
                "chart_json": chart_json,
                "charts_json": charts_json,
                "data": display_data,
                "suggestions": suggestions,
                "insights": insights,
                "reasoning": reasoning_steps,
            }

        # Exhausted iterations — return what we have
        return {
            "answer": "I've been processing your request but couldn't reach a final answer. Please try rephrasing your question.",
            "chart_json": None,
            "charts_json": [],
            "data": None,
            "suggestions": None,
            "insights": None,
            "reasoning": reasoning_steps,
        }

    # ── Internal: chart extraction ───────────────────────────

    def _extract_and_build_charts(
        self, answer: str, data: list[dict[str, Any]]
    ) -> list[str]:
        """Extract all chart specs from the LLM's answer and build Plotly figures.

        Looks for ```chart ... ``` code blocks in the answer containing
        JSON objects with chart_type, x, y, etc.
        """
        # Primary: look for ```chart ... ``` fenced blocks
        pattern = r"```chart\s*\n?\s*(\{.*?\})\s*\n?\s*```"
        matches_list = list(re.finditer(pattern, answer, re.DOTALL))

        # Fallback: if no fenced block, look for a bare JSON object with chart_type
        if not matches_list:
            bare_pattern = r'(\{\s*"chart_type"\s*:.+?\})'
            matches_list = list(re.finditer(bare_pattern, answer, re.DOTALL))

        matches = iter(matches_list)

        charts_json: list[str] = []

        # If tool_data is empty (e.g. get_summary_statistics returns a dict),
        # fall back to the full dataset so histograms / distributions can render
        if not data:
            from backend.dataset import get_dataframe
            data = get_dataframe().to_dict(orient="records")
            if not data:
                return charts_json

        for match in matches:
            try:
                spec = json.loads(match.group(1))
            except json.JSONDecodeError:
                logger.warning("Failed to parse chart spec JSON from LLM answer.")
                continue

            chart_type = spec.get("chart_type", "bar")
            x = spec.get("x", "")
            y = spec.get("y", "")
            title = spec.get("title", "")
            color = spec.get("color")

            # Map common LLM naming variations/synonyms to actual dataset column names
            synonyms = {
                "country": "Area",
                "Country": "Area",
                "crop": "Item",
                "Crop": "Item",
                "yield": "hg/ha_yield",
                "Yield": "hg/ha_yield",
                "rainfall": "average_rain_fall_mm_per_year",
                "Rainfall": "average_rain_fall_mm_per_year",
                "temp": "avg_temp",
                "Temp": "avg_temp",
                "temperature": "avg_temp",
                "Temperature": "avg_temp",
                "pesticide": "pesticides_tonnes",
                "Pesticide": "pesticides_tonnes",
                "pesticides": "pesticides_tonnes",
                "Pesticides": "pesticides_tonnes",
                "year": "Year",
            }
            if x in synonyms:
                x = synonyms[x]
            if y in synonyms:
                y = synonyms[y]
            if color in synonyms:
                color = synonyms[color]

            # Histograms only need x; all other chart types require both x and y
            charts_needing_y = {"bar", "line", "scatter", "area", "box"}
            if not x or (not y and chart_type in charts_needing_y):
                continue

            try:
                fig = create_chart(
                    chart_type=chart_type,
                    data=data,
                    x=x,
                    y=y,
                    title=title,
                    color=color,
                )
                charts_json.append(figure_to_json(fig))
            except Exception as e:
                logger.error(f"Chart generation error: {e}")

        return charts_json

    def _extract_suggestions(self, answer: str) -> list[str] | None:
        """Extract follow-up question suggestions from the LLM's answer.

        Looks for bullet points under a "Try Asking" heading.
        """
        # Look for the suggestions section
        pattern = r"(?:Try Asking|Suggested|Follow-up)[:\s]*\n((?:\s*[-*]\s+.+\n?)+)"
        match = re.search(pattern, answer, re.IGNORECASE)

        if not match:
            return None

        # Extract individual bullet points
        bullets = re.findall(r"[-*]\s+(.+)", match.group(1))
        suggestions = [b.strip().strip("*").strip() for b in bullets if b.strip()]

        return suggestions[:5] if suggestions else None

    def _auto_generate_chart(
        self,
        reasoning_steps: list[dict],
        tool_data: list[dict],
        answer_text: str,
    ) -> list[str]:
        """Automatically generate a sensible chart when the LLM omits a chart spec.

        Picks chart type and columns based on which tool was called and what
        data is available, so visualizations always appear even when the model
        doesn't output a ```chart``` block.
        """
        # Find the last tool call name and arguments
        tool_calls = [s for s in reasoning_steps if s.get("type") == "tool_call"]
        if not tool_calls:
            return []

        last_call = tool_calls[-1]
        tool_name = last_call.get("name", "")
        tool_args = last_call.get("arguments", {})

        if not tool_name:
            return []

        # Resolve what data to use (tools now provide the 'data' subset natively)
        data = tool_data
        if not data:
            return []

        spec: dict = {}

        # ── Map tool → chart spec ─────────────────────────────────────────
        if tool_name == "get_summary_statistics":
            column = tool_args.get("column", "hg/ha_yield")
            label_map = {
                "avg_temp": "Average Temperature (°C)",
                "hg/ha_yield": "Crop Yield (hg/ha)",
                "average_rain_fall_mm_per_year": "Rainfall (mm/year)",
                "pesticides_tonnes": "Pesticide Usage (tonnes)",
            }
            
            # If the LLM is discussing a time trend, aggregate and plot a line chart
            ans_lower = answer_text.lower()
            if "year" in ans_lower or "trend" in ans_lower or "over time" in ans_lower:
                df = pd.DataFrame(data)
                if "Year" in df.columns:
                    # Aggregate by year for a smooth trend line
                    df_agg = df.groupby("Year")[column].mean().reset_index()
                    data = df_agg.to_dict(orient="records")
                    spec = {
                        "chart_type": "line",
                        "x": "Year",
                        "y": column,
                        "title": f"Trend of {label_map.get(column, column)} Over Years",
                    }
                else:
                    spec = {
                        "chart_type": "histogram",
                        "x": column,
                        "title": f"Distribution of {label_map.get(column, column)} Across All Countries",
                    }
            else:
                spec = {
                    "chart_type": "histogram",
                    "x": column,
                    "title": f"Distribution of {label_map.get(column, column)} Across All Countries",
                }

        elif tool_name in ("filter_by_country", "get_yield_trend"):
            country = tool_args.get("country", "")
            item = tool_args.get("item")
            
            # Default to crop yield bar chart
            metric = "hg/ha_yield"
            chart_title = f"Average Crop Yield in {country}"

            if item:
                spec = {
                    "chart_type": "line",
                    "x": "Year",
                    "y": metric,
                    "title": f"{item} {chart_title}",
                    "color": "Item",
                }
            else:
                spec = {
                    "chart_type": "bar",
                    "x": "Item",
                    "y": "hg/ha_yield",
                    "title": chart_title,
                    "color": "Item",
                }

        elif tool_name == "filter_by_item":
            item = tool_args.get("item", "Crop")
            spec = {
                "chart_type": "bar",
                "x": "Area",
                "y": "hg/ha_yield",
                "title": f"{item} Yield by Country",
                "color": "Area",
            }

        elif tool_name == "compare_countries":
            item = tool_args.get("item", "")
            metric = tool_args.get("metric", "hg/ha_yield")
            
            label_map = {
                "avg_temp": "Average Temperature (°C)",
                "hg/ha_yield": "Yield (hg/ha)",
                "average_rain_fall_mm_per_year": "Rainfall (mm/year)",
                "pesticides_tonnes": "Pesticide Usage (tonnes)",
            }
            metric_label = label_map.get(metric, metric)
            
            # If only one country is actually present in data (other was missing from dataset),
            # adjust the title to reflect this partial result
            _df_check = pd.DataFrame(data)
            available_countries = _df_check["Area"].unique().tolist() if "Area" in _df_check.columns else []
            if len(available_countries) == 1:
                title = f"{item} {metric_label} in {available_countries[0]} (Partial — other country not in dataset)" if item else f"{metric_label} in {available_countries[0]} (Partial — other country not in dataset)"
            else:
                title = f"{item} {metric_label} Comparison: {' vs '.join(available_countries)}" if item else f"{metric_label} Comparison: {' vs '.join(available_countries)}"
            
            spec = {
                "chart_type": "line",
                "x": "Year",
                "y": metric,
                "title": title,
                "color": "Area",
            }

        elif tool_name == "get_top_countries":
            metric = tool_args.get("metric", "hg/ha_yield")
            item = tool_args.get("item", "")
            year = tool_args.get("year", "")
            
            label_map = {
                "avg_temp": "Average Temperature (°C)",
                "hg/ha_yield": "Yield (hg/ha)",
                "average_rain_fall_mm_per_year": "Rainfall (mm/year)",
                "pesticides_tonnes": "Pesticide Usage (tonnes)",
            }
            metric_label = label_map.get(metric, metric)
            
            title = f"Top Countries by {metric_label}"
            if item:
                title = f"Top {item} Producers"
            if year:
                title += f" in {year}"
                
            spec = {
                "chart_type": "bar",
                "x": "Area",
                "y": metric,
                "title": title,
                "color": "Area",
            }

        elif tool_name == "get_correlation":
            col_x = tool_args.get("col_x", "avg_temp")
            col_y = tool_args.get("col_y", "hg/ha_yield")
            spec = {
                "chart_type": "scatter",
                "x": col_x,
                "y": col_y,
                "title": f"Correlation: {col_x} vs {col_y}",
            }

        elif tool_name == "get_pesticide_usage":
            country = tool_args.get("country") or "All Countries"
            spec = {
                "chart_type": "line",
                "x": "Year",
                "y": "pesticides_tonnes",
                "title": f"Pesticide Usage Over Time — {country}",
                "color": "Area",
            }

        elif tool_name == "forecast_yield_trend":
            country = tool_args.get("country", "")
            item = tool_args.get("item", "")
            spec = {
                "chart_type": "line",
                "x": "Year",
                "y": "hg/ha_yield",
                "title": f"{item} Yield Forecast in {country}",
                "color": "type",
            }

        if not spec:
            return []

        # Build the chart
        x = spec.get("x", "")
        y = spec.get("y", "")
        chart_type = spec.get("chart_type", "bar")
        charts_needing_y = {"bar", "line", "scatter", "area", "box"}
        if not x or (not y and chart_type in charts_needing_y):
            return []

        try:
            fig = create_chart(
                chart_type=chart_type,
                data=data,
                x=x,
                y=y or None,
                title=spec.get("title", ""),
                color=spec.get("color"),
            )
            return [figure_to_json(fig)]
        except Exception as e:
            import traceback
            with open("chart_error.txt", "w") as f:
                f.write(traceback.format_exc())
            logger.error(f"Auto-chart generation error: {e}")
            return []
