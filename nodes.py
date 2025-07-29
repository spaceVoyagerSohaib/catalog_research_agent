import logging
from langchain_core.messages import HumanMessage
from models import ResearchState, SearchAttempt, VerificationResult
from config import Config
from prompt_loader import prompt_loader
from tools import initial_search
from tools import deep_search

logger = logging.getLogger(__name__)

class Nodes:
    def __init__(self, llm):
        self.llm = llm
        self.verification_llm = llm.with_structured_output(VerificationResult)
    
    def _check_duplicate_search(self, search_history: list, component: str) -> bool:
        return any(attempt.get('query') == component for attempt in search_history)
    
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
            
            if self._check_duplicate_search(search_history, component):
                logger.warning("Duplicate search detected")
                return {'termination_reason': 'duplicate_search'}
            
            query = self._generate_search_query(component)
            logger.debug(f"LLM generated query: {query}")
            
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
    
    def _categorize_sources(self, analysis, state: ResearchState) -> tuple:
        verified_sources = state.get('verified_sources', [])
        failed_sources = state.get('failed_sources', [])
        
        all_sources = (
            analysis.description_sources + 
            analysis.active_date_sources + 
            analysis.eos_date_sources
        )
        
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
            
            analysis_prompt = prompt_loader.get_prompt(
                "verification", 
                "analysis", 
                component=state.get('component'),
                raw_content=raw_content
            )
            
            logger.debug(f"Verification LLM prompt: {analysis_prompt}")
            
            analysis = self.verification_llm.invoke([HumanMessage(content=analysis_prompt)])
            logger.debug(f"Verification LLM output: {analysis}")
            logger.debug(f"Verification confidence: {analysis.overall_confidence}")
            
            verified_sources, failed_sources = self._categorize_sources(analysis, state)
            
            return {
                'current_results': analysis.dict(exclude={'overall_confidence', 'verification_notes'}),
                'confidence_score': analysis.overall_confidence,
                'verified_sources': verified_sources,
                'failed_sources': failed_sources,
                'verification_notes': analysis.verification_notes
            }
            
        except Exception as e:
            logger.error(f"Verification failed: {str(e)}")
            return {
                'confidence_score': 0.0,
                'termination_reason': f'error: {str(e)}',
                'iteration_count': state.get('iteration_count', 0) + 1
            }
    
    def _generate_followup_query(self, component: str) -> str:
        return f"{component} lifecycle support dates official documentation"
    
    def _execute_deep_search(self, query: str):
        return deep_search.invoke({'query': query})
    
    def followup_research_node(self, state: ResearchState) -> ResearchState:
        logger.debug(f"Followup research node input state: {state}")
        logger.info(f"Starting follow-up research for component: {state.get('component', 'Unknown')}")
        
        try:
            component = state.get('component')
            followup_query = self._generate_followup_query(component)
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
        confidence = state.get('confidence_score', 0.0)
        iterations = state.get('iteration_count', 0)
        logger.info(f"Decision: confidence={confidence}, iterations={iterations}")
        
        if confidence >= 90.0 or iterations >= 2:
            return "output_generation"
        else:
            return "followup_research"
 
    def _create_error_output(self, state: ResearchState) -> dict:
        return {
            'component': state.get('component'),
            'description': 'Information not available',
            'active_date': 'Unknown',
            'eos_date': 'Unknown',
            'confidence_score': 0.0,
            'sources': {
                'description_sources': [],
                'active_date_sources': [],
                'eos_date_sources': []
            },
            'verification_notes': state.get('verification_notes', 'No verification performed'),
            'iteration_count': state.get('iteration_count', 0),
            'error': 'Unable to extract reliable information from search results'
        }
    
    def _create_successful_output(self, state: ResearchState, current_results: dict) -> dict:
        return {
            'component': state.get('component'),
            'description': current_results.get('description'),
            'active_date': current_results.get('active_date'),
            'eos_date': current_results.get('eos_date'),
            'confidence_score': state.get('confidence_score', 0.0),
            'sources': {
                'description_sources': current_results.get('description_sources', []),
                'active_date_sources': current_results.get('active_date_sources', []),
                'eos_date_sources': current_results.get('eos_date_sources', [])
            },
            'verification_notes': state.get('verification_notes', ''),
            'iteration_count': state.get('iteration_count', 0)
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