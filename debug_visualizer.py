import sys
import json
import logging
from typing import Dict, Any, List
from config import Config
from nodes import Nodes
from models import ResearchState
from main import load_components


def _truncate(text: str, limit: int = 1200) -> str:
    if not text:
        return ""
    text = text.strip()
    return text


def _print_rule():
    print("-" * 80)


def _print_box(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _show_research_io(nodes: Nodes, input_state: Dict[str, Any], output_state: Dict[str, Any]):
    _print_box("Node: research")
    print("Input:")
    print(f"- component: {input_state.get('component')}")
    print(f"- prior_searches: {len(input_state.get('search_history', []))}")
    _print_rule()
    print("Output:")
    sh: List[Dict[str, Any]] = output_state.get("search_history", [])
    if sh:
        attempt = sh[-1]
        results = attempt.get("results") or {}
        print(f"- query: {attempt.get('query')}")
        print(f"- mode: {attempt.get('mode')}")
        print(f"- model: {results.get('model')}")
        print("- raw_content:")
        print(_truncate(results.get("raw_content") or ""))


def _show_verification_io(nodes: Nodes, input_state: Dict[str, Any], output_state: Dict[str, Any]):
    _print_box("Node: verification")
    # Input: last raw_content used for verification
    try:
        raw = nodes._extract_search_content(input_state.get("search_history", []))  # noqa: SLF001
    except Exception:
        raw = ""
    print("Input:")
    print("- raw_content:")
    print(_truncate(raw))
    _print_rule()
    print("Output:")
    cr = output_state.get("current_results") or {}
    print(f"- active_date: {cr.get('active_date')}  status={cr.get('status_active')}  conf={cr.get('confidence_active')}")
    print(f"- eos_date:    {cr.get('eos_date')}  status={cr.get('status_eos')}  conf={cr.get('confidence_eos')}")
    print(f"- verified_sources: {len(output_state.get('verified_sources', []))}")
    print(f"- failed_sources:   {len(output_state.get('failed_sources', []))}")
    print(f"- confidence_score: {output_state.get('confidence_score')}")


def _show_followup_io(nodes: Nodes, input_state: Dict[str, Any], output_state: Dict[str, Any]):
    _print_box("Node: followup_research")
    print("Input:")
    curr = input_state.get("current_results") or {}
    print(f"- prev_status_active={curr.get('status_active')} prev_status_eos={curr.get('status_eos')}")
    _print_rule()
    print("Output:")
    sh: List[Dict[str, Any]] = output_state.get("search_history", [])
    if sh:
        attempt = sh[-1]
        results = attempt.get("results") or {}
        print(f"- followup_query: {attempt.get('query')}")
        print(f"- mode: {attempt.get('mode')}")
        print(f"- model: {results.get('model')}")
        print("- raw_content:")
        print(_truncate(results.get("raw_content") or ""))


def _show_decision(next_node: str):
    _print_box("Decision")
    print(f"Next: {next_node}")


def _show_output(input_state: Dict[str, Any], output_state: Dict[str, Any]):
    _print_box("Node: output_generation")
    print(f"Input: current_results present = {input_state.get('current_results') is not None}")
    _print_rule()
    print("Output:")
    out = output_state.get("output") or {}
    print(json.dumps(out, indent=2))


def visualize_component(component: str):
    llm = Config.get_llm()
    nodes = Nodes(llm)

    state: ResearchState = {
        "component": component,
        "search_history": [],
        "current_results": None,
        "confidence_score": 0.0,
        "iteration_count": 0,
        "verified_sources": [],
        "failed_sources": [],
        "termination_reason": None,
        "verification_notes": None,
        "output": None,
    }

    # research
    research_in = dict(state)
    research_out = nodes.research_node(state)
    state.update(research_out)
    _show_research_io(nodes, research_in, research_out)

    if state.get("termination_reason"):
        _print_box("Terminated")
        print(state.get("termination_reason"))
        return

    # verification
    ver_in = dict(state)
    ver_out = nodes.verification_node(state)
    state.update(ver_out)
    _show_verification_io(nodes, ver_in, ver_out)

    # loop (decision -> followup_research -> verification) up to 2 iterations
    for _ in range(2):
        next_node = nodes.decision_node(state)
        _show_decision(next_node)
        if next_node == "output_generation":
            out_in = dict(state)
            out_out = nodes.output_generation_node(state)
            state.update(out_out)
            _show_output(out_in, out_out)
            break
        else:
            fu_in = dict(state)
            fu_out = nodes.followup_research_node(state)
            state.update(fu_out)
            _show_followup_io(nodes, fu_in, fu_out)

            ver_in = dict(state)
            ver_out = nodes.verification_node(state)
            state.update(ver_out)
            _show_verification_io(nodes, ver_in, ver_out)


def main():
    Config.load()
    logging.disable(logging.CRITICAL)
    if len(sys.argv) < 2:
        print("Usage: python catalog_research_agent/debug_visualizer.py <components_file>")
        sys.exit(1)

    components = load_components(sys.argv[1])

    for comp in components:
        print("\n" + "#" * 80)
        print(f"Component: {comp}")
        print("#" * 80)
        visualize_component(comp)


if __name__ == "__main__":
    main() 