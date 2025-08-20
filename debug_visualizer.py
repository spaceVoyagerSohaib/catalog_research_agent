import sys
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from config import Config
from nodes import Nodes
from models import ResearchState
from main import load_components


@dataclass
class VerificationImpact:
    component: str
    initial_active_date: Optional[str]
    initial_eos_date: Optional[str]
    initial_confidence_active: float
    initial_confidence_eos: float
    initial_status_active: str
    initial_status_eos: str
    
    final_active_date: Optional[str]
    final_eos_date: Optional[str]
    final_confidence_active: float
    final_confidence_eos: float
    final_status_active: str
    final_status_eos: str
    final_confidence_score: float
    
    had_followup: bool
    followup_query: Optional[str]
    iterations: int
    
    # Impact metrics
    active_date_discovered: bool  # Found date when initial had none
    eos_date_discovered: bool
    active_confidence_improved: float  # Delta in confidence
    eos_confidence_improved: float
    status_refined: bool  # Status changed from ambiguous to verified
    contradiction_detected: bool  # Different dates found


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


def visualize_component(component: str) -> VerificationImpact:
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
        return None

    # Initial verification - capture baseline metrics
    ver_in = dict(state)
    ver_out = nodes.verification_node(state)
    state.update(ver_out)
    _show_verification_io(nodes, ver_in, ver_out)
    
    # Capture initial results after first verification
    initial_results = state.get("current_results") or {}
    initial_active_date = initial_results.get('active_date')
    initial_eos_date = initial_results.get('eos_date')
    initial_confidence_active = initial_results.get('confidence_active', 0.0) or 0.0
    initial_confidence_eos = initial_results.get('confidence_eos', 0.0) or 0.0
    initial_status_active = initial_results.get('status_active', 'not_found')
    initial_status_eos = initial_results.get('status_eos', 'not_found')

    had_followup = False
    followup_query = None
    iterations = state.get('iteration_count', 0)

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
            had_followup = True
            fu_in = dict(state)
            fu_out = nodes.followup_research_node(state)
            state.update(fu_out)
            _show_followup_io(nodes, fu_in, fu_out)
            
            # Capture follow-up query
            if not followup_query:
                search_history = state.get('search_history', [])
                if search_history:
                    followup_query = search_history[-1].get('query')

            ver_in = dict(state)
            ver_out = nodes.verification_node(state)
            state.update(ver_out)
            _show_verification_io(nodes, ver_in, ver_out)

    # Capture final results
    final_output = state.get("output") or {}
    final_results = state.get("current_results") or {}
    
    final_active_date = final_output.get('active_date')
    final_eos_date = final_output.get('eos_date')
    final_confidence_active = final_results.get('confidence_active', 0.0) or 0.0
    final_confidence_eos = final_results.get('confidence_eos', 0.0) or 0.0
    final_status_active = final_results.get('status_active', 'not_found')
    final_status_eos = final_results.get('status_eos', 'not_found')
    final_confidence_score = state.get('confidence_score', 0.0)
    final_iterations = state.get('iteration_count', 0)

    # Calculate impact metrics
    active_date_discovered = (initial_active_date is None and final_active_date is not None)
    eos_date_discovered = (initial_eos_date is None and final_eos_date is not None)
    active_confidence_improved = final_confidence_active - initial_confidence_active
    eos_confidence_improved = final_confidence_eos - initial_confidence_eos
    status_refined = (
        (initial_status_active == 'ambiguous' and final_status_active == 'verified') or
        (initial_status_eos == 'ambiguous' and final_status_eos == 'verified')
    )
    contradiction_detected = (
        (initial_active_date is not None and final_active_date is not None and initial_active_date != final_active_date) or
        (initial_eos_date is not None and final_eos_date is not None and initial_eos_date != final_eos_date)
    )

    return VerificationImpact(
        component=component,
        initial_active_date=initial_active_date,
        initial_eos_date=initial_eos_date,
        initial_confidence_active=initial_confidence_active,
        initial_confidence_eos=initial_confidence_eos,
        initial_status_active=initial_status_active,
        initial_status_eos=initial_status_eos,
        
        final_active_date=final_active_date,
        final_eos_date=final_eos_date,
        final_confidence_active=final_confidence_active,
        final_confidence_eos=final_confidence_eos,
        final_status_active=final_status_active,
        final_status_eos=final_status_eos,
        final_confidence_score=final_confidence_score,
        
        had_followup=had_followup,
        followup_query=followup_query,
        iterations=final_iterations,
        
        active_date_discovered=active_date_discovered,
        eos_date_discovered=eos_date_discovered,
        active_confidence_improved=active_confidence_improved,
        eos_confidence_improved=eos_confidence_improved,
        status_refined=status_refined,
        contradiction_detected=contradiction_detected
    )


def _generate_verification_impact_report(impacts: List[VerificationImpact]) -> str:
    """Generate comprehensive verification impact analysis report."""
    valid_impacts = [i for i in impacts if i is not None]
    total_components = len(valid_impacts)
    
    if total_components == 0:
        return "No valid components processed."
    
    # Calculate aggregate metrics
    components_with_followup = sum(1 for i in valid_impacts if i.had_followup)
    active_dates_discovered = sum(1 for i in valid_impacts if i.active_date_discovered)
    eos_dates_discovered = sum(1 for i in valid_impacts if i.eos_date_discovered)
    status_refinements = sum(1 for i in valid_impacts if i.status_refined)
    contradictions_detected = sum(1 for i in valid_impacts if i.contradiction_detected)
    
    # Confidence improvements
    active_conf_improvements = [i.active_confidence_improved for i in valid_impacts if i.active_confidence_improved > 0]
    eos_conf_improvements = [i.eos_confidence_improved for i in valid_impacts if i.eos_confidence_improved > 0]
    
    avg_active_improvement = sum(active_conf_improvements) / len(active_conf_improvements) if active_conf_improvements else 0
    avg_eos_improvement = sum(eos_conf_improvements) / len(eos_conf_improvements) if eos_conf_improvements else 0
    
    # Build report
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("CHAIN-OF-VERIFICATION IMPACT ANALYSIS")
    lines.append("=" * 80)
    lines.append(f"Total Components Analyzed: {total_components}")
    lines.append(f"Components with Follow-up Research: {components_with_followup} ({100*components_with_followup/total_components:.1f}%)")
    lines.append("")
    
    lines.append("VERIFICATION VALUE-ADD METRICS:")
    lines.append("-" * 40)
    lines.append(f"• Active Dates Discovered: {active_dates_discovered} ({100*active_dates_discovered/total_components:.1f}%)")
    lines.append(f"• EOS Dates Discovered: {eos_dates_discovered} ({100*eos_dates_discovered/total_components:.1f}%)")
    lines.append(f"• Status Refinements: {status_refinements} ({100*status_refinements/total_components:.1f}%)")
    lines.append(f"• Contradictions Detected: {contradictions_detected} ({100*contradictions_detected/total_components:.1f}%)")
    lines.append("")
    
    lines.append("CONFIDENCE IMPROVEMENTS:")
    lines.append("-" * 40)
    lines.append(f"• Components with Active Confidence Boost: {len(active_conf_improvements)} (avg: +{avg_active_improvement:.1f}%)")
    lines.append(f"• Components with EOS Confidence Boost: {len(eos_conf_improvements)} (avg: +{avg_eos_improvement:.1f}%)")
    lines.append("")
    
    # Top examples of verification value
    lines.append("TOP VERIFICATION SUCCESS STORIES:")
    lines.append("-" * 40)
    
    # Sort by most impactful
    impact_scores = []
    for i in valid_impacts:
        score = 0
        if i.active_date_discovered: score += 3
        if i.eos_date_discovered: score += 3
        if i.contradiction_detected: score += 2
        if i.status_refined: score += 1
        score += (i.active_confidence_improved + i.eos_confidence_improved) / 20  # Scale confidence improvements
        impact_scores.append((score, i))
    
    impact_scores.sort(key=lambda x: x[0], reverse=True)
    
    for score, impact in impact_scores[:5]:  # Top 5 examples
        if score > 0:
            lines.append(f"• {impact.component}")
            improvements = []
            if impact.active_date_discovered:
                improvements.append(f"Found active date: {impact.final_active_date}")
            if impact.eos_date_discovered:
                improvements.append(f"Found EOS date: {impact.final_eos_date}")
            if impact.contradiction_detected:
                improvements.append("Detected contradiction")
            if impact.status_refined:
                improvements.append("Refined status")
            if impact.active_confidence_improved > 5:
                improvements.append(f"Active confidence +{impact.active_confidence_improved:.1f}%")
            if impact.eos_confidence_improved > 5:
                improvements.append(f"EOS confidence +{impact.eos_confidence_improved:.1f}%")
            
            lines.append(f"  → {', '.join(improvements)}")
            if impact.followup_query:
                lines.append(f"  → Follow-up query: {impact.followup_query[:100]}...")
            lines.append("")
    
    # Components where verification made no difference
    no_impact = [i for i in valid_impacts if not i.had_followup]
    if no_impact:
        lines.append(f"COMPONENTS WITH NO FOLLOW-UP NEEDED: {len(no_impact)}")
        lines.append("-" * 40)
        for impact in no_impact[:3]:  # Show first 3
            lines.append(f"• {impact.component} - Initial verification sufficient")
        if len(no_impact) > 3:
            lines.append(f"  ... and {len(no_impact) - 3} more")
        lines.append("")
    
    return "\n".join(lines)


def main():
    Config.load()
    logging.disable(logging.CRITICAL)
    if len(sys.argv) < 2:
        print("Usage: python debug_visualizer.py <components_file>")
        sys.exit(1)

    components = load_components(sys.argv[1])
    impacts = []

    for comp in components:
        print("\n" + "#" * 80)
        print(f"Component: {comp}")
        print("#" * 80)
        impact = visualize_component(comp)
        impacts.append(impact)

    # Generate and display verification impact analysis
    impact_report = _generate_verification_impact_report(impacts)
    print(impact_report)


if __name__ == "__main__":
    main()
