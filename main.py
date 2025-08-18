import asyncio
import json
import time
from typing import List, Dict, Any
from datetime import datetime
from config import Config
from graph import CatalogResearchGraph

class CatalogResearchRunner:
    def __init__(self, max_concurrent: int = 3):
        self.max_concurrent = max_concurrent
        self.graph = None
    
    def _init_graph(self):
        if not self.graph:
            Config.load()
            self.graph = CatalogResearchGraph().build()
    
    async def _process_component(self, component: str, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        async with semaphore:
            self._init_graph()
            
            config = {"configurable": {"thread_id": f"batch_{component}_{int(time.time())}"}}
            input_state = {
                "component": component,
                "search_history": [],
                "current_results": None,
                "confidence_score": 0.0,
                "iteration_count": 0,
                "verified_sources": [],
                "failed_sources": [],
                "termination_reason": None,
                "verification_notes": None
            }
            
            try:
                start_time = time.time()
                final_state = await asyncio.to_thread(self.graph.invoke, input_state, config)
                processing_time = time.time() - start_time
                
                result = {
                    "component": component,
                    "processing_time": processing_time,
                    "status": "completed",
                    "timestamp": datetime.now().isoformat()
                }
                
                if 'output' in final_state and final_state['output']:
                    result.update(final_state['output'])
                else:
                    result.update({
                        "status": "failed",
                        "error": "No output generated",
                        "termination_reason": final_state.get('termination_reason', 'unknown')
                    })
                
                return result
                
            except Exception as e:
                return {
                    "component": component,
                    "status": "failed",
                    "error": str(e),
                    "processing_time": time.time() - start_time,
                    "timestamp": datetime.now().isoformat()
                }
    
    async def run_batch(self, components: List[str]) -> Dict[str, Any]:
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        print(f"🔍 Catalog Research Agent - Processing {len(components)} components...")
        start_time = time.time()
        
        tasks = [self._process_component(comp, semaphore) for comp in components]
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        successful = len([r for r in results if r.get('status') == 'completed'])
        
        return {
            "batch_metadata": {
                "total_components": len(components),
                "successful": successful,
                "failed": len(components) - successful,
                "processing_time": total_time,
                "timestamp": datetime.now().isoformat()
            },
            "results": results
        }
    
    def export_json(self, batch_results: Dict[str, Any], fpath: str = './outputs', filename: str = None) -> str:
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"catalog_research_results_{timestamp}.json"
        
        with open(fpath + '/' + filename, 'w') as f:
            json.dump(batch_results, f, indent=4)
        
        return filename

def load_components(file_path: str) -> List[str]:
    with open(file_path, 'r') as f:
        if file_path.endswith('.json'):
            data = json.load(f)
            return data if isinstance(data, list) else data.get('components', [])
        else:
            return [line.strip() for line in f if line.strip()]

async def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python main.py <components_file> [max_concurrent]")
        print("\nCatalog Research Agent - Software Lifecycle Research Automation")
        print("=" * 60)
        print("Example: python main.py components.txt 3")
        sys.exit(1)
    
    components_file = sys.argv[1]
    max_concurrent = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    
    print("Catalog Research Agent - Software Lifecycle Research")
    print("=" * 60)
    
    components = load_components(components_file)
    runner = CatalogResearchRunner(max_concurrent)
    
    batch_results = await runner.run_batch(components)
    filename = runner.export_json(batch_results)
    
    print("\n BATCH RESULTS")
    print("=" * 60)
    print(f" Completed: {batch_results['batch_metadata']['successful']}/{batch_results['batch_metadata']['total_components']}")
    print(f" Processing time: {batch_results['batch_metadata']['processing_time']:.2f} seconds")
    print(f" Results exported to: {filename}")

if __name__ == "__main__":
    asyncio.run(main()) 