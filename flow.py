"""
================================================================================
PORTABLE TUTORIAL GENERATOR - FLOW DEFINITION
================================================================================
This file defines the PocketFlow workflow that orchestrates the tutorial 
generation process. It's an EXACT COPY of the main flow.py with added comments.

FLOW ARCHITECTURE:
==================
The tutorial generation follows a 6-node pipeline:

    FetchRepo → IdentifyAbstractions → AnalyzeRelationships → 
    OrderChapters → WriteChapters → CombineTutorial

Each node follows the PocketFlow pattern:
    - prep(shared): Read data from shared store, prepare for processing
    - exec(prep_res): Execute the main logic (usually LLM calls)
    - post(shared, prep_res, exec_res): Write results back to shared store

The ">>" operator connects nodes, creating a linear flow.
When a node completes, the next node in the chain is executed.

IMPORTANT: This file must stay in sync with the main project's flow.py!
================================================================================
"""

from pocketflow import Flow

# Import all node classes from nodes.py
# Each class handles one step in the tutorial generation pipeline
from nodes import (
    FetchRepo,              # Step 1: Fetch files from local directory or GitHub
    IdentifyAbstractions,   # Step 2: Use LLM to identify core concepts
    AnalyzeRelationships,   # Step 3: Use LLM to find how concepts relate
    OrderChapters,          # Step 4: Use LLM to determine teaching order
    WriteChapters,          # Step 5: Use LLM to write each chapter (BatchNode)
    CombineTutorial         # Step 6: Combine all chapters into final output
)


def create_tutorial_flow():
    """
    Creates and returns the codebase tutorial generation flow.
    
    This function instantiates all nodes with their retry configurations
    and connects them in sequence to form the complete pipeline.
    
    RETRY CONFIGURATION:
    - max_retries=5: Each node will attempt up to 5 times on failure
    - wait=20: Wait 20 seconds between retries (helps with rate limits)
    
    FetchRepo doesn't need retries since it's not calling an LLM.
    CombineTutorial doesn't need retries since it's just file I/O.
    
    Returns:
        Flow: A PocketFlow Flow object ready to be run with shared data
    """

    # ==========================================================================
    # STEP 1: Instantiate all nodes
    # ==========================================================================
    
    # FetchRepo: Reads files from local directory or GitHub repository
    # No retries needed - just file system operations
    fetch_repo = FetchRepo()
    
    # IdentifyAbstractions: Uses LLM to analyze codebase and find core concepts
    # Needs retries because LLM calls can fail or return invalid YAML
    identify_abstractions = IdentifyAbstractions(max_retries=5, wait=20)
    
    # AnalyzeRelationships: Uses LLM to understand how abstractions interact
    # Needs retries for same reasons as above
    analyze_relationships = AnalyzeRelationships(max_retries=5, wait=20)
    
    # OrderChapters: Uses LLM to determine optimal teaching order
    # Needs retries for same reasons as above
    order_chapters = OrderChapters(max_retries=5, wait=20)
    
    # WriteChapters: Uses LLM to write each chapter (this is a BatchNode!)
    # BatchNode means it processes multiple items - one per chapter
    # The exec() method is called once per chapter in the order
    write_chapters = WriteChapters(max_retries=5, wait=20)
    
    # CombineTutorial: Combines all chapters into final Markdown files
    # No retries needed - just file I/O operations
    combine_tutorial = CombineTutorial()

    # ==========================================================================
    # STEP 2: Connect nodes in sequence using ">>" operator
    # ==========================================================================
    # The ">>" operator creates a "default" transition between nodes.
    # When node A's post() returns "default" (or None), it moves to node B.
    
    fetch_repo >> identify_abstractions
    # After fetching files, identify the core abstractions
    
    identify_abstractions >> analyze_relationships
    # After identifying abstractions, analyze how they relate
    
    analyze_relationships >> order_chapters
    # After understanding relationships, determine chapter order
    
    order_chapters >> write_chapters
    # After ordering, write each chapter (BatchNode processes all chapters)
    
    write_chapters >> combine_tutorial
    # After all chapters written, combine into final tutorial

    # ==========================================================================
    # STEP 3: Create and return the Flow
    # ==========================================================================
    # The Flow starts with fetch_repo and follows the chain we defined above
    tutorial_flow = Flow(start=fetch_repo)

    return tutorial_flow
