import logging
from langchain_core.messages import HumanMessage
from models import ResearchState, SearchAttempt, ActiveVerificationResult, EosVerificationResult
from config import Config
from prompt_loader import prompt_loader
from tools import initial_search
from tools import deep_search

logger = logging.getLogger(__name__)

class Nodes:
    def __init__(self, llm):
        self.llm = llm
        self.llm_json = llm.bind(response_format={"type": "json_object"})
        self.active_verification_llm = self.llm_json.with_structured_output(ActiveVerificationResult)
        self.eos_verification_llm = self.llm_json.with_structured_output(EosVerificationResult)
    
    def _check_duplicate_search(self, search_history: list, query: str) -> bool:
        return any(attempt.get('query') == query for attempt in search_history)
    
    def _generate_search_query(self, component: str) -> str:
        query_prompt = prompt_loader.get_prompt("research", "query_generation", component=component)
        logger.debug(f"LLM query generation prompt: {query_prompt}")
        
        query_msg = self.llm.invoke([HumanMessage(content=query_prompt)])
        query = query_msg.content.strip()
        
        return self._clean_query(query)
    
    def _clean_query(self, query: str) -> str:
        if '\n' in query:
            query = query.split('\n')[0].strip()
        if query.startswith('"') and query.endswith('"'):
            query = query[1:-1]
        if query.startswith("Query:"):
            query = query[6:].strip()
        return query
    
    def _execute_search(self, query: str):
        return initial_search.invoke({'query': query})
    
    def research_node(self, state: ResearchState) -> ResearchState:
        logger.debug(f"Research node input state: {state}")
        logger.info(f"Starting research for component: {state.get('component', 'Unknown')}")
        
        try:
            search_history = state.get('search_history', [])
            component = state.get('component')

            query = self._generate_search_query(component)
            logger.debug(f"LLM generated query: {query}")

            if self._check_duplicate_search(search_history, query):
                logger.warning("Duplicate search detected")
                return {'termination_reason': 'duplicate_search'}
            
            tool_result = self._execute_search(query)
            
            updated_history = search_history + [SearchAttempt(
                query=query, 
                mode='sonar-pro', 
                results=tool_result, 
                confidence=0.0
            )]
            
            return {
                'search_history': updated_history,
                'iteration_count': state.get('iteration_count', 0) + 1
            }
            
        except Exception as e:
            logger.error(f"Research node failed: {str(e)}")
            return {
                'termination_reason': f'error: {str(e)}',
                'iteration_count': state.get('iteration_count', 0) + 1
            }
    
    def _extract_search_content(self, search_history: list) -> str:
        if not search_history:
            raise ValueError("No search history available for verification")
        
        last_search = search_history[-1]
        search_results = last_search.get('results', {})
        return search_results.get('raw_content', '')
    
    def _categorize_sources(self, active_sources, eos_sources, state: ResearchState) -> tuple:
        verified_sources = state.get('verified_sources', [])
        failed_sources = state.get('failed_sources', [])
        
        all_sources = []
        if active_sources:
            all_sources.extend(active_sources)
        if eos_sources:
            all_sources.extend(eos_sources)

        for source in all_sources:
            if source.credibility_score >= 70:
                verified_sources.append(source.url)
            else:
                failed_sources.append(source.url)
        
        return verified_sources, failed_sources
    
    def verification_node(self, state: ResearchState) -> ResearchState:
        logger.debug(f"Verification node input state: {state}")
        logger.info(f"Verifying results for iteration {state.get('iteration_count', 0)}")
        
        try:
            search_history = state.get('search_history', [])
            raw_content = self._extract_search_content(search_history)
            component = state.get('component')

            active_prompt = prompt_loader.get_prompt(
                "verification_active",
                "analysis",
                component=component,
                raw_content=raw_content
            )
            eos_prompt = prompt_loader.get_prompt(
                "verification_eos",
                "analysis",
                component=component,
                raw_content=raw_content
            )

            logger.debug(f"Active verification prompt: {active_prompt}")
            active = self.active_verification_llm.invoke([HumanMessage(content=active_prompt)])

            logger.debug(f"EOS verification prompt: {eos_prompt}")
            eos = self.eos_verification_llm.invoke([HumanMessage(content=eos_prompt)])

            combined_result = {
                'active_date': active.active_date,
                'active_date_sources': [s.dict() for s in (active.active_date_sources or [])],
                'eos_date': eos.eos_date,
                'eos_date_sources': [s.dict() for s in (eos.eos_date_sources or [])],
                'confidence_active': active.confidence_active,
                'confidence_eos': eos.confidence_eos,
                'status_active': active.status_active,
                'status_eos': eos.status_eos,
                'notes_active': active.notes_active,
                'notes_eos': eos.notes_eos,
            }

            # Simple confidence score (use minimum like original overall_confidence concept)
            confidence_score = min(active.confidence_active, eos.confidence_eos)
            
            verified_sources, failed_sources = self._categorize_sources(
                active.active_date_sources or [], 
                eos.eos_date_sources or [], 
                state
            )
            
            return {
                'current_results': combined_result,
                'confidence_score': confidence_score,
                'verified_sources': verified_sources,
                'failed_sources': failed_sources,
            }
            
        except Exception as e:
            logger.error(f"Verification failed: {str(e)}")
            return {
                'confidence_score': 0.0,
                'termination_reason': f'error: {str(e)}',
                'iteration_count': state.get('iteration_count', 0) + 1
            }
    
    def _generate_followup_query(self, state: ResearchState) -> str:
        component = state.get('component', '')
        current = state.get('current_results') or {}
        
        if (current.get('status_active') == 'verified' and 
            current.get('status_eos') != 'verified'):
            return f'{component} "end of support" OR "EOL" OR "end of life"'
    
        return f"{component} lifecycle support dates"
    
    def _execute_deep_search(self, query: str):
        return deep_search.invoke({'query': query})
    
    def followup_research_node(self, state: ResearchState) -> ResearchState:
        logger.debug(f"Followup research node input state: {state}")
        logger.info(f"Starting follow-up research for component: {state.get('component', 'Unknown')}")
        
        try:
            followup_query = self._generate_followup_query(state)
            logger.debug(f"Follow-up query: {followup_query}")
 
            tool_result = self._execute_deep_search(followup_query)
 
            search_history = state.get('search_history', [])
            updated_history = search_history + [SearchAttempt(
                query=followup_query,
                mode='deep',
                results=tool_result,
                confidence=0.0
            )]
            
            return {
                'search_history': updated_history,
                'iteration_count': state.get('iteration_count', 0) + 1
            }
            
        except Exception as e:
            logger.error(f"Follow-up research failed: {str(e)}")
            return {
                'termination_reason': f'error: {str(e)}',
                'iteration_count': state.get('iteration_count', 0) + 1
            }
 
    def decision_node(self, state: ResearchState) -> str:
        logger.debug(f"Decision node input state: {state}")
        iterations = state.get('iteration_count', 0)
        current_results = state.get('current_results') or {}
        conf_active = current_results.get('confidence_active', 0.0) or 0.0
        conf_eos = current_results.get('confidence_eos', None)
        status_active = current_results.get('status_active', 'not_found')
        status_eos = current_results.get('status_eos', 'not_found')

        active_threshold = 85.0
        eos_threshold = 85.0

        active_ok = (status_active == "verified" and conf_active >= active_threshold)
        eos_ok = (
            (status_eos in {"verified", "derived"} and (conf_eos or 0.0) >= eos_threshold)
            or status_eos == "not_applicable"
        )

        logger.info(f"Decision: iterations={iterations}, active_ok={active_ok}, eos_ok={eos_ok}")

        if (active_ok and eos_ok) or iterations >= 2:
            return "output_generation"
        else:
            return "followup_research"
 
    def _create_error_output(self, state: ResearchState) -> dict:
        return {
            'component': state.get('component'),
            'active_date': 'Unknown',
            'eos_date': 'Unknown',
            'confidence_score': 0.0,
            'sources': {
                'active_date_sources': [],
                'eos_date_sources': []
            },
            'verification_notes': state.get('verification_notes', 'No verification performed'),
            'iteration_count': state.get('iteration_count', 0),
            'confidence_active': 0.0,
            'confidence_eos': 0.0,
            'status_active': 'not_found',
            'status_eos': 'not_found',
            'error': 'Unable to extract reliable information from search results'
        }
    
    def _create_successful_output(self, state: ResearchState, current_results: dict) -> dict:
        return {
            'component': state.get('component'),
            'active_date': current_results.get('active_date'),
            'eos_date': current_results.get('eos_date'),
            'confidence_score': state.get('confidence_score', 0.0),
            'sources': {
                'active_date_sources': current_results.get('active_date_sources', []),
                'eos_date_sources': current_results.get('eos_date_sources', [])
            },
            'notes_active': current_results.get('notes_active', ''),
            'notes_eos': current_results.get('notes_eos', ''),
            'iteration_count': state.get('iteration_count', 0),
            # Per-field additions
            'confidence_active': current_results.get('confidence_active'),
            'confidence_eos': current_results.get('confidence_eos'),
            'status_active': current_results.get('status_active'),
            'status_eos': current_results.get('status_eos'),
        }
    
    def output_generation_node(self, state: ResearchState) -> ResearchState:
        logger.debug(f"Output generation node input state: {state}")
        logger.info("Generating final output")
        
        try:
            current_results = state.get('current_results')
            
            if current_results is None:
                logger.warning("No current_results available, creating basic output")
                output = self._create_error_output(state)
            else:
                output = self._create_successful_output(state, current_results)
            
            logger.debug(f"Generated output: {output}")
            return {'output': output, 'termination_reason': 'completed'}
            
        except Exception as e:
            logger.error(f"Output generation failed: {str(e)}")
            return {'termination_reason': f'error: {str(e)}'}
